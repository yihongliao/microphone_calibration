import numpy as np
from scipy.io.wavfile import write

def generate_sweep_signal(duration, start_freq, end_freq, sample_rate, filename):
    """
    Generates a sweep signal from start_freq to end_freq over the given duration
    and saves it as a wave file.

    :param duration: Duration of the sweep signal in seconds
    :param start_freq: Starting frequency of the sweep (in Hz)
    :param end_freq: Ending frequency of the sweep (in Hz)
    :param sample_rate: Sample rate (in samples per second)
    :param filename: Name of the output wave file
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    sweep = np.sin(2 * np.pi * (start_freq * t + (end_freq - start_freq) / (2 * duration) * t**2))
    # Normalize to 16-bit PCM range
    sweep = np.int16(sweep / np.max(np.abs(sweep)) * 32767)
    write(filename, sample_rate, sweep)

# Parameters
duration = 10  # 10 seconds
start_freq = 0  # 0 Hz
end_freq = 20000  # 20000 Hz
sample_rate = 44100  # CD quality sample rate
filename = '../signal/frequency_sweep.wav'

generate_sweep_signal(duration, start_freq, end_freq, sample_rate, filename)
print(f"Sweep signal saved as {filename}")
