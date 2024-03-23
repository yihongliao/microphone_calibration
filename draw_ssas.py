import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks

def draw_ssas(channel_data, fs):

    L = len(channel_data)

    # Compute the one-dimensional discrete Fourier Transform
    spectrum = np.fft.fft(channel_data) / L

    # Calculate the frequencies corresponding to the spectrum
    frequencies = np.fft.fftfreq(L, d=1/fs)

    # Keep only the positive frequencies (single-side spectrum)
    positive_frequencies = frequencies[:L//2]
    
    positive_spectrum = np.abs(spectrum[:L//2])
    positive_spectrum[1:-1] = 2*positive_spectrum[1:-1]


    # Plot the single-side amplitude spectrum
    # plt.figure(figsize=(5, 5))
    plt.plot(positive_frequencies, positive_spectrum)
    plt.xlim(0, fs/2)
    plt.ylim(0, 0.0004)
    plt.title('Single-side Amplitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    # plt.show()

def plot_single_side_amplitude_spectrum(audio_file):
    # Read audio file
    sample_rate, data = wavfile.read(audio_file)
    if data.ndim > 1:
        channel_data = data[:,0] / np.iinfo(np.int16).max
    else:
        channel_data = data / np.iinfo(np.int16).max

    L = len(channel_data)

    # Compute the one-dimensional discrete Fourier Transform
    spectrum = np.fft.fft(channel_data) / L

    # Calculate the frequencies corresponding to the spectrum
    frequencies = np.fft.fftfreq(L, d=1/sample_rate)

    # Keep only the positive frequencies (single-side spectrum)
    positive_frequencies = frequencies[:L//2]
    
    positive_spectrum = np.abs(spectrum[:L//2])
    positive_spectrum[1:-1] = 2*positive_spectrum[1:-1]


    # Plot the single-side amplitude spectrum
    plt.figure(figsize=(5, 5))
    plt.plot(positive_frequencies, positive_spectrum)
    plt.xlim(0, 20000)
    plt.ylim(0, 0.0004)
    plt.title('Single-side Amplitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Example usage
# audio_file_path = '../measurements/ambient_w_arm1.wav'
# plot_single_side_amplitude_spectrum(audio_file_path)
