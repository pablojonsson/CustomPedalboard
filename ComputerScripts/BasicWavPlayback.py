import sounddevice as sd
import soundfile as sf

filename = "P:\CustomPedalboard\AudioSamples\CleanGuitar.wav "
data, samplerate = sf.read(filename, dtype="float32")

sd.play(data, samplerate)
sd.wait()
