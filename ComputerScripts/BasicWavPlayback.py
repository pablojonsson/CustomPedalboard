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


def main():
    FILENAME_IN = "P:\\CustomPedalboard\\AudioSamples\\CleanGuitar.wav"
    FILENAME_OUT = "P:\\CustomPedalboard\\AudioSamples\\modified.wav"

    data, samplerate = sf.read(FILENAME_IN, dtype="float32")

    try:
        delay_time = float(input("Enter delay time in seconds: "))
        feedback = float(input("Enter feedback amount: "))
        mix = float(input("Enter dry/wet mix (0.0 = dry, 1.0 = wet): "))
    except ValueError:
        print("Invalid input. Using default delay parameters.")
        delay_time = .3
        feedback = .5
        mix = .5

    print(
        f"Applying Delay: Time={delay_time}s, Feedback={feedback}, Mix={mix}")

    # CODE FOR apply_gain
    '''try:
        gain_db = float(input("Enter the desired gain in dB: "))
    except ValueError:
        print("Invalid input. Using default gain of 0 dB.")
        gain_db = 0.0

    processed_data = apply_gain(data, gain_db)'''

    processed_data = apply_delay(data, samplerate, delay_time, feedback, mix)

    sf.write(FILENAME_OUT, processed_data, samplerate)

    player = AudioPlayer()

    print("Playing audio... Type 'stop' to stop, 'quit' to exit.")
    player.play_audio(processed_data, samplerate)

    try:
        while player.is_playing:
            user_input = input("> ").lower().strip()
            if user_input == 'stop':
                player.stop()
                print("Playback stopped.")
            elif user_input == 'quit':
                player.stop()
                break

            time.sleep(.1)
    except KeyboardInterrupt:
        print("\nStopping playback...")
        player.stop()


if __name__ == "__main__":
    main()
