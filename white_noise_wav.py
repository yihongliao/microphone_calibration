import numpy as np
from scipy.io.wavfile import write

def generate_white_noise_with_silence(duration_noise, duration_silence, sample_rate, filename):
    """
    Generates a white noise signal with a specified duration of silence at the beginning
    and saves it as a wave file.

    :param duration_noise: Duration of the white noise in seconds
    :param duration_silence: Duration of the silence in seconds
    :param sample_rate: Sample rate (in samples per second)
    :param filename: Name of the output wave file
    """
    # Generate silence
    silence = np.zeros(int(sample_rate * duration_silence), dtype=np.int16)
    
    # Generate white noise
    noise = np.random.normal(0, 1, int(sample_rate * duration_noise))
    # Normalize to 16-bit PCM range
    noise = np.int16(noise / np.max(np.abs(noise)) * 32767)
    
    # Concatenate silence and noise
    signal = np.concatenate((silence, noise))
    
    # Write to wave file
    write(filename, sample_rate, signal)

# Parameters
duration_noise = 2  # 10 seconds of white noise
duration_silence = 0  # 10 seconds of silence at the beginning
sample_rate = 44100  # CD quality sample rate
filename = f"../signal/white_noise_with_{duration_silence}s_silence.wav"

generate_white_noise_with_silence(duration_noise, duration_silence, sample_rate, filename)
print(f"White noise with silence saved as {filename}")
