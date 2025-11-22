import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time

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

# double comb filter


def two_parallel_combs(input_signal, samplerate, delay_time1=.03, feedback1=.7, delay_time2=.037, feedback2=.7):
    if input_signal.ndim > 1:
        input_signal = np.mean(input_signal, axis=1)

    output = np.zeros_like(input_signal)

    M1 = int(delay_time1 * samplerate)
    M2 = int(delay_time2 * samplerate)

    buffer1 = np.zeros(M1)
    buffer2 = np.zeros(M2)
    idx1, idx2 = 0, 0

    for n in range(len(input_signal)):
        current_input = input_signal[n]

        delayed1 = buffer1[idx1]
        out1 = current_input + feedback1 * delayed1
        buffer1[idx1] = current_input + feedback1 * delayed1
        idx1 = (idx1 + 1) % M1

        delayed2 = buffer2[idx2]
        out2 = current_input + feedback2 * delayed2
        buffer2[idx2] = current_input + feedback2 * delayed2
        idx2 = (idx2 + 1) % M2

        output[n] = (out1 + out2) / 2.0

    return output
# applies a digital reverb to the given signal using several parallel comb filters
# all fed into two all pass filters in series.


def apply_reverb(input_signal, samplerate, delay_time=.03, feedback=.7):
    output_signal = two_parallel_combs(input_signal, samplerate)
    return output_signal


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
            mix = float(input("Enter drt/wet mix (0.0 = dry, 1.0 = wet): "))
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
            delay_time = float(input("Enter delay time in seconds: "))
            feedback = float(input("Enter feedback amount (0.0 - 1.0): "))
        except ValueError:
            print("Invalid input, Using default reverb parameters")
            delay_time = .03
            feedback = .7
        params['delay_time'] = delay_time
        params['feedback'] = feedback

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
    FILENAME_IN = "P:\\CustomPedalboard\\AudioSamples\\CleanGuitar.wav"
    FILENAME_OUT = "P:\\CustomPedalboard\\AudioSamples\\processed_chain.wav"

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
    print("  'quit' - Stop playback and exit")
    print("  Ctrl+C - Stop playback and exit")
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
