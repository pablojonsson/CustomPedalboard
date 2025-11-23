import numpy as np
import sounddevice as sd
import threading
import time
from collections import deque


class RealTimeEffectProcessor:
    def __init__(self, samplerate=44100, buffer_size=256):
        self.samplerate = samplerate
        self.buffer_size = buffer_size
        self.is_running = False
        self.audio_thread = None

        self.effects_chain = []
        self.current_params = {}

        self.delay_buffers = {}
        self.filter_states = {}
        self.reverb_tails = None

    def add_effect(self, effect_func, params):
        self.effects_chain.append(effect_func)
        self.current_params.update(params)

    def process_buffer(self, input_buffer):
        current_signal = input_buffer

        for effect in self.effects_chain:
            if __name__ == 'real_time_gain':
                current_signal = self.real_time_gain(current_signal)
            elif effect.__name__ == 'real_time_delay':
                current_signal = self.real_time_delay(current_signal)

        return current_signal

    def real_time_gain(self, buffer):
        gain_linear = 10 ** (self.current_params.get('gain_db', 0) / 20)
        return buffer * gain_linear

    def real_time_delay(self, buffer):
        delay_samples = int(self.current_params.get(
            'delay_time', .3) * self.samplerate)
        feedback = self.current_params.get('feedback', .5)
        mix = self.current_params.get('mix', .5)

        if 'delay_buffer' not in self.delay_buffers:
            self.delay_buffers['delay_buffer'] = np.zeros(delay_samples)
            self.delay_buffers['delay_index'] = 0

        delay_buf = self.delay_buffers['delay_buffer']
        buf_idx = self.delay_buffers['delay_index']

        output_buffer = np.zeros_like(buffer)

        for i in range(len(buffer)):
            input_sample = buffer[i]

            delayed_sample = delay_buf[buf_idx]

            output_sample = (1 - mix) * input_sample + mix * delayed_sample
            output_buffer[i] = output_sample

            delay_buf[buf_idx] = input_sample + feedback * delayed_sample

            buf_idx = (buf_idx + 1) % len(delay_buf)

        self.delay_buffers['delay_index'] = buf_idx

        return output_buffer

    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Audio status: {status}")

        processed = self.process_buffer(indata[:, 0])

        outdata[:, 0] = processed
        if outdata.shape[1] > 1:
            outdata[:, 1] = processed

    def start_live_processing(self):
        self.is_running = True
        self.stream = sd.Stream(
            samplerate=self.samplerate,
            blocksize=self.buffer_size,
            channels=1,
            dtype='float32',
            callback=self.audio_callback
        )

        self.stream.start()

    def stop_processing(self):
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()


def main():
    processor = RealTimeEffectProcessor(samplerate=44100, buffer_size=256)

    print("Starting real-time processing...")
    print("Speak into your microphone - you should hear processed audio")
    print("Press Enter to stop...")
    processor.start_live_processing()

    try:
        input()
    except KeyboardInterrupt:
        pass
    finally:
        processor.stop_processing()
        print("Stopped real-time processing")


if __name__ == "__main__":
    main()
