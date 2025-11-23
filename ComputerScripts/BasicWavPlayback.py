import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time
import random


# AudioPlayer class with flag variables for if the audio is playing and a
# reference to the Thread object.


class AudioPlayer:
    def __init__(self):
        self.is_playing = False
        self.playback_thread = None

    # play audio in a separate thread

    def play_audio(self, data, samplerate):
        def _play():
            self.is_playing = True
            sd.play(data, samplerate)
            # blocks thread, but not the program
            sd.wait()
            self.is_playing = False

        self.playback_thread = threading.Thread(target=_play)
        self.playback_thread.daemon = True
        self.playback_thread.start()

    # stop playback

    def stop(self):
        sd.stop()
        self.is_playing = False

    # wait for playback to complete

    def wait_until_done(self):
        if self.playback_thread:
            self.playback_thread.join()


# applies a gain change to a given signal
def apply_gain(input_signal, gain_db):
    gain_linear = 10 ** (gain_db / 20)  # db are on a logarithmic scale
    print(
        f"Applying {gain_db} dB of gain (linear multiplier: {gain_linear:.4f})")

    output_signal = input_signal * gain_linear

    output_signal = np.clip(output_signal, -1.0, 1.0)

    return output_signal


# applies a digital delay to a given signal using a circular buffer
def apply_delay(input_signal, samplerate, delay_time, feedback, mix=.5):

    if input_signal.ndim == 1:
        num_channels = 1
        input_signal = input_signal[:, np.newaxis]
    else:
        num_channels = input_signal.shape[1]

    num_samples = input_signal.shape[0]
    delay_samples = int(delay_time * samplerate)

    buffer_length = delay_samples + num_samples
    delay_buffer = np.zeros(
        (buffer_length, num_channels), dtype=input_signal.dtype)

    output_signal = np.zeros_like(input_signal)

    for i in range(num_samples):
        for ch in range(num_channels):
            input_sample = input_signal[i, ch]

            read_index = i - delay_samples
            if read_index >= 0:
                delayed_sample = delay_buffer[read_index, ch]
            else:
                delayed_sample = 0.0

            dry = (1.0 - mix) * input_sample
            wet = mix * delayed_sample
            output_sample = dry + wet

            delay_buffer[i, ch] = input_sample + (feedback * delayed_sample)

            output_signal[i, ch] = output_sample

    if num_channels == 1:
        output_signal = output_signal[:, 0]

    return output_signal


def generate_comb_delays(target_time, spread=.005, samplerate=44100):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def are_mutually_prime(numbers):
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                if gcd(numbers[i], numbers[j]) != 1:
                    return False

        return True

    target_samples = int(target_time * samplerate)
    spread_samples = int(spread * samplerate)

    min_samples = target_samples - spread_samples
    max_samples = target_samples + spread_samples

    min_samples = max(10, min_samples)
    max_samples = min(20000, max_samples)

    max_attempts = 500

    for attempt in range(max_attempts):
        candidates = []
        for _ in range(4):
            candidate = random.randint(min_samples, max_samples)
            candidates.append(candidate)

        if are_mutually_prime(candidates):
            delay_times = [samples / samplerate for samples in candidates]
            delay_times.sort()

            return delay_times

    base_delays = [1320, 1636, 1813, 1928]
    scale_factor = target_time / .0371

    delay_times = [(delay * scale_factor) /
                   samplerate for delay in base_delays]

    return delay_times


# double comb filter
def four_parallel_combs(input_signal, samplerate, delay_times=[0.0297, 0.0371, 0.0411, 0.0437], feedbacks=[.7, .7, .7, .7]):

    if input_signal.ndim > 1:
        input_signal = np.mean(input_signal, axis=1)

    output = np.zeros_like(input_signal)

    num_combs = len(delay_times)

    buffers = []
    indices = []
    delays_samples = []

    for i in range(num_combs):
        M = int(delay_times[i] * samplerate)
        delays_samples.append(M)
        buffers.append(np.zeros(M))
        indices.append(0)

    for n in range(len(input_signal)):

        current_input = input_signal[n]
        comb_sum = 0.0

        for i in range(num_combs):
            M = delays_samples[i]
            g = feedbacks[i]
            buf = buffers[i]
            idx = indices[i]

            delayed = buf[idx]

            comb_out = current_input + g * delayed

            buf[idx] = current_input + g * delayed

            indices[i] = (idx + 1) % M

            comb_sum += comb_out

        output[n] = comb_sum / num_combs

    return output

# applies a digital reverb to the given signal using several parallel comb filters
# all fed into two all pass filters in series.


def allpass_filter(input_signal, delay_time=.005, gain=.5, samplerate=44100):
    if input_signal.ndim > 1:
        input_signal = np.mean(input_signal, axis=1)

    output = np.zeros_like(input_signal)
    M = int(delay_time * samplerate)

    if M == 0:
        return input_signal

    delay_buffer = np.zeros(M)
    buffer_index = 0

    for n in range(len(input_signal)):
        x = input_signal[n]
        delayed = delay_buffer[buffer_index]

        y = -gain * x + delayed
        if n > 0:
            y += gain * output[n-1]

        delay_buffer[buffer_index] = x + gain * delayed

        buffer_index = (buffer_index + 1) % M
        output[n] = y

    return output


def apply_reverb(input_signal, samplerate, delay_time=.03, feedback=.7, rt60=2.0, mix=.5):
    if input_signal.ndim > 1:
        input_signal = np.mean(input_signal, axis=1)

    output = np.zeros_like(input_signal)

    comb_delays = generate_comb_delays(
        delay_time, spread=.005, samplerate=samplerate)

    comb_feedbacks = []
    for M_samples in [int(d * samplerate) for d in comb_delays]:
        g = 10 ** (-3 * M_samples / (rt60 * samplerate))
        comb_feedbacks.append(g * .9)

    wet_signal = four_parallel_combs(
        input_signal, samplerate, comb_delays, comb_feedbacks)

    wet_gain = 1.0 / (1.0 + (mix * rt60 * .3))
    wet_signal = wet_signal * wet_gain

    wet_signal = allpass_filter(
        wet_signal, delay_time=.005, gain=.5, samplerate=samplerate)
    wet_signal = allpass_filter(
        wet_signal, delay_time=.0017, gain=.5, samplerate=samplerate)

    dry_signal = input_signal
    output = (1-mix) * dry_signal + mix * wet_signal

    peak_before_clip = np. max(np.abs(output))
    if peak_before_clip > .9:
        output = np.tanh(output * .95)

    return output


EFFECTS_REGISTRY = {
    'gain': apply_gain,
    'delay': apply_delay,
    'reverb': apply_reverb
}


def get_effect_parameters(effect_name):
    params = {}

    if effect_name == 'gain':
        try:
            gain_db = float(input("Enter gain in dB: "))
        except ValueError:
            print("Invalid input. Using default gain of 0dB.")
            gain_db = 0.0
        params['gain_db'] = gain_db

    elif effect_name == 'delay':
        try:
            delay_time = float(input("Enter delay time in seconds: "))
            feedback = float(input("Enter feedback amount (0.0-1.0): "))
            mix = float(input("Enter dry/wet mix (0.0 = dry, 1.0 = wet): "))
        except ValueError:
            print("Invalid input. Using default delay parameters")
            delay_time = .3
            feedback = .5
            mix = .5
        params['delay_time'] = delay_time
        params['feedback'] = feedback
        params['mix'] = mix

    elif effect_name == 'reverb':
        try:
            delay_time = float(input("Enter base delay time in seconds: "))
            rt60 = float(input("Enter reveb decay time in seconds: "))
            mix = float(input("Enter dry/wet mix: "))
        except ValueError:
            print("Invalid input, Using default reverb parameters")
            delay_time = .03
            rt60 = 2.0
            mix = .5
        params['delay_time'] = delay_time
        params['rt60'] = rt60
        params['mix'] = mix

    # more elif statements for more effects

    return params


def build_signal_chain():
    signal_chain = []

    print("\n" + "="*50)
    print("BUILD YOUR PEDALBOARD")
    print("="*50)
    print(f"Available effects: {', '.join(EFFECTS_REGISTRY.keys())}")
    print("Type the name of an effect to add it to your chain.")
    print("Type 'done' when finished building your chain.")
    print("Type 'quit' to exit the program.")
    print("Type 'list' to see current chain.")
    print("Type 'remove' to remove last effect.")
    print("-" * 50)

    while True:
        user_input = input("\nAdd effect > ").lower().strip()

        if user_input == 'done':
            if not signal_chain:
                print("No effects in chain. Using dry signal.")
            break
        elif user_input == 'quit':
            return None
        elif user_input == 'list':
            if signal_chain:
                chain_names = [func.__name__.replace(
                    'apply_', '') for func, _ in signal_chain]
                print(f"Current chain: {' -> '.join(chain_names)}")
            else:
                print("Chain is empty.")
        elif user_input == 'remove':
            if signal_chain:
                removed_effect = signal_chain.pop()
                print(
                    f"Removed {removed_effect[0].__name__.replace('apply_', '')}")
                if signal_chain:
                    chain_names = [func.__name__.replace(
                        'apply_', '') for func, _ in signal_chain]
                    print(f"Current chain: {' -> '.join(chain_names)}")
                else:
                    print("Chain is now empty.")
            else:
                print("No effects to remove.")
        elif user_input in EFFECTS_REGISTRY:
            effect_func = EFFECTS_REGISTRY[user_input]
            print(f"\nConfiguring {user_input}...")

            params = get_effect_parameters(user_input)

            signal_chain.append((effect_func, params))
            print(f"Added {user_input} to the signal chain.")

            chain_names = [func.__name__.replace(
                'apply_', '') for func, _ in signal_chain]
            print(f"Current chain: {' -> '.join(chain_names)}")
            print("Add another effect or type 'done' to finish.")
        else:
            print(
                f"Unknown effect '{user_input}'. Available effects: {', '.join(EFFECTS_REGISTRY.keys())}")
            print(
                "Type 'list' to see current chain, 'done' to finish, or 'quit' to exit.")

    return signal_chain


def main():
    FILENAME_IN = "P:/CustomPedalboard/AudioSamples/CleanGuitar.wav"
    FILENAME_OUT = "P:/CustomPedalboard/AudioSamples/processed_chain.wav"

    data, samplerate = sf.read(FILENAME_IN, dtype="float32")

    # Build the signal chain interactively
    signal_chain = build_signal_chain()
    if signal_chain is None:  # User typed 'quit'
        print("Exiting program.")
        return

    # Process the audio through the complete chain
    print("\n" + "="*50)
    print("PROCESSING AUDIO")
    print("="*50)

    if not signal_chain:
        print("No effects selected. Playing dry signal.")
        processed_data = data
    else:
        chain_names = [func.__name__.replace(
            'apply_', '') for func, _ in signal_chain]
        print(f"Processing through chain: {' -> '.join(chain_names)}")
        processed_data = process_audio_chain(data, samplerate, signal_chain)

    # Save and play the result
    print("\nSaving processed audio...")
    sf.write(FILENAME_OUT, processed_data, samplerate)

    player = AudioPlayer()
    print("\n" + "="*50)
    print("PLAYBACK CONTROLS")
    print("="*50)
    print("Playing processed audio...")
    print("Commands during playback:")
    print("  'stop' - Stop playback and exit")
    player.play_audio(processed_data, samplerate)

    try:
        while player.is_playing:
            user_input = input("> ").lower().strip()
            if user_input in ['stop', 'quit']:
                player.stop()
                print("Playback stopped.")
                break
            else:
                print("Unknown command. Type 'stop' to stop playback.")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping playback...")
        player.stop()


def process_audio_chain(input_signal, samplerate, signal_chain):
    current_signal = input_signal

    for i, (effect_func, params) in enumerate(signal_chain):
        print(f"Applying {effect_func.__name__.replace('apply_', '')}...")

        try:
            if effect_func == apply_delay or effect_func == apply_reverb:
                current_signal = effect_func(
                    current_signal, samplerate, **params)
            else:
                current_signal = effect_func(current_signal, **params)

        except Exception as e:
            print(f"Error applying {effect_func.__name__}: {e}")
            print("Continuing with previous signal...")

    return current_signal


if __name__ == "__main__":
    main()
