import numpy as np
import sounddevice as sd
import threading
import time


class LowLatencyEffectProcessor:
    def __init__(self, samplerate=44100, buffer_size=128):
        self.samplerate = samplerate
        self.buffer_size = buffer_size
        self.is_running = False

        # Effect chain
        self.effects_chain = []
        self.current_params = {}

        # State persistence
        self.delay_buffers = {}
        self.filter_states = {}

        # Latency measurement
        self.callback_times = []
        self.processing_times = []

    def add_effect(self, effect_func, params):
        self.effects_chain.append(effect_func)
        self.current_params.update(params)

    def process_buffer(self, input_buffer):
        """Optimized buffer processing"""
        start_process = time.time()

        current_signal = input_buffer

        for effect in self.effects_chain:
            current_signal = effect(current_signal)

        # Measure processing time
        process_time = (time.time() - start_process) * 1000  # Convert to ms
        self.processing_times.append(process_time)

        return current_signal

    def real_time_gain(self, buffer):
        gain_linear = 10 ** (self.current_params.get('gain_db', 0) / 20)
        return buffer * gain_linear

    def real_time_delay(self, buffer):
        delay_time = self.current_params.get('delay_time', 0.3)
        feedback = self.current_params.get('feedback', 0.5)
        mix = self.current_params.get('mix', 0.5)

        delay_samples = int(delay_time * self.samplerate)

        # Initialize delay buffer if needed
        if 'delay_buffer' not in self.delay_buffers:
            self.delay_buffers['delay_buffer'] = np.zeros(delay_samples)
            self.delay_buffers['delay_index'] = 0

        delay_buf = self.delay_buffers['delay_buffer']
        buf_idx = self.delay_buffers['delay_index']

        output_buffer = np.zeros_like(buffer)

        # Process buffer (this is the bottleneck)
        for i in range(len(buffer)):
            input_sample = buffer[i]
            delayed_sample = delay_buf[buf_idx]

            output_buffer[i] = (1 - mix) * input_sample + mix * delayed_sample
            delay_buf[buf_idx] = input_sample + feedback * delayed_sample
            buf_idx = (buf_idx + 1) % len(delay_buf)

        self.delay_buffers['delay_index'] = buf_idx
        return output_buffer

    def audio_callback(self, indata, outdata, frames, time_info, status):
        callback_start = time.time()

        if status:
            print(f"Audio status: {status}")

        # Process buffer
        input_buffer = indata[:, 0]
        processed = self.process_buffer(input_buffer)

        # Output
        outdata[:, 0] = processed
        if outdata.shape[1] > 1:
            outdata[:, 1] = processed

        # Measure total callback time
        callback_time = (time.time() - callback_start) * 1000
        self.callback_times.append(callback_time)

    def print_latency_stats(self):
        """Print latency statistics"""
        if self.callback_times:
            avg_callback = np.mean(
                self.callback_times[-100:])  # Last 100 calls
            max_callback = np.max(self.callback_times[-100:])
            avg_process = np.mean(self.processing_times[-100:])

            buffer_latency_ms = (self.buffer_size / self.samplerate) * 1000

            print(f"\n--- Latency Statistics ---")
            print(f"Buffer latency: {buffer_latency_ms:.1f} ms")
            print(f"Avg callback time: {avg_callback:.2f} ms")
            print(f"Max callback time: {max_callback:.2f} ms")
            print(f"Avg processing time: {avg_process:.2f} ms")
            print(
                f"Total estimated latency: {buffer_latency_ms + avg_callback:.1f} ms")

            # Warning if we're missing deadlines
            if max_callback > buffer_latency_ms:
                print("⚠️  WARNING: Processing may miss real-time deadlines!")

    def start_live_processing(self):
        """Start real-time audio processing"""
        print(f"Starting real-time processing")
        print(f"Buffer size: {self.buffer_size} samples")
        print(f"Sample rate: {self.samplerate} Hz")
        print(
            f"Theoretical latency: {(self.buffer_size / self.samplerate) * 1000:.1f} ms")
        print("\nSpeak into your microphone - you should hear processed audio")
        print("Press Enter to stop...")

        self.is_running = True

        try:
            # This is the key real-time audio stream
            self.stream = sd.Stream(
                samplerate=self.samplerate,
                blocksize=self.buffer_size,
                channels=1,  # Mono input
                dtype='float32',
                callback=self.audio_callback
            )

            self.stream.start()

        except Exception as e:
            print(f"Error starting audio stream: {e}")
            print(
                "Make sure you have a microphone connected and Python sounddevice installed")

    def stop_processing(self):
        """Stop real-time processing"""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()


def test_different_buffer_sizes():
    """Test different buffer sizes to find optimal latency"""
    buffer_sizes = [64, 128, 256, 512]

    for buffer_size in buffer_sizes:
        print(f"\n{'='*50}")
        print(f"Testing buffer size: {buffer_size}")
        print(f"{'='*50}")

        processor = LowLatencyEffectProcessor(
            samplerate=44100,
            buffer_size=buffer_size
        )

        # Simple effect chain for testing
        processor.add_effect(processor.real_time_gain, {'gain_db': 6.0})

        print("Starting processing... Speak into microphone")
        print("Press Enter after 5 seconds to continue...")

        processor.start_live_processing()

        # Run for 5 seconds to collect stats
        time.sleep(5)

        processor.stop_processing()
        processor.print_latency_stats()


def main():
    """Main function with optimal settings"""
    # Start with a small buffer size
    processor = LowLatencyEffectProcessor(samplerate=44100, buffer_size=128)

    # Use only lightweight effects for low latency
    processor.add_effect(processor.real_time_gain, {'gain_db': 6.0})
    # processor.add_effect(processor.real_time_delay, {  # Comment out for minimum latency
    #     'delay_time': 0.1,
    #     'feedback': 0.3,
    #     'mix': 0.2
    # })

    print("Starting LOW LATENCY processing...")
    print("Buffer size: 128 samples (2.9ms theoretical)")
    print("Speak into microphone - you should feel less latency")
    print("Press Enter to stop...")

    # Start a background thread to print stats
    def print_stats():
        while processor.is_running:
            time.sleep(2)
            processor.print_latency_stats()

    stats_thread = threading.Thread(target=print_stats, daemon=True)

    processor.start_live_processing()
    stats_thread.start()

    try:
        input()
    except KeyboardInterrupt:
        pass
    finally:
        processor.stop_processing()
        print("\nFinal statistics:")
        processor.print_latency_stats()


if __name__ == "__main__":
    # Uncomment to test different buffer sizes
    # test_different_buffer_sizes()

    # Run the main low-latency version
    main()
