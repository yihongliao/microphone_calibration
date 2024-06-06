import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.fft import fft
from scipy.ndimage import gaussian_filter1d

def smooth_octave(y, freqs, fraction=1/3):
    """
    Smooth the amplitude response in a fractional octave band.
    
    :param y: The amplitude spectrum to be smoothed
    :param freqs: The corresponding frequencies
    :param fraction: The fraction of octave to use for smoothing
    :return: The smoothed amplitude spectrum
    """
    smoothed_y = np.zeros_like(y)
    for i, f in enumerate(freqs):
        if f == 0:
            smoothed_y[i] = y[i]
        else:
            octave_band = (2**(-fraction/2) * f, 2**(fraction/2) * f)
            indices = np.where((freqs >= octave_band[0]) & (freqs <= octave_band[1]))[0]
            smoothed_y[i] = np.mean(y[indices])
    return smoothed_y


def plot_frequency_response(filename, channels):
    """
    Reads a wave file containing a sweep signal and plots its frequency response on a logarithmic scale.
    Applies 1/3 octave smoothing to the frequency response.
    
    :param filename: Name of the input wave file containing the sweep signal
    :param channels: List of channel indices to plot
    """
    # Read the wave file
    sample_rate, data = read(filename)
    
    # If the data has more than one channel, extract the specified channels
    if len(data.shape) == 1:
        raise ValueError("The file is not a multi-channel wave file.")
    
    plt.figure(figsize=(12, 6))
    
    for channel in channels:
        # Extract the channel data
        channel_data = data[:, channel]
        
        # Number of samples
        n = len(channel_data)
        
        # Perform Fourier Transform
        yf = fft(channel_data)
        
        # Frequency bins
        xf = np.linspace(0.0, sample_rate / 2.0, n // 2)
        
        # Compute the magnitude in dB
        magnitude = 2.0 / n * np.abs(yf[:n // 2])
        magnitude_db = 20 * np.log10(magnitude)
        
        # Apply 1/3 octave smoothing
        smoothed_magnitude_db = smooth_octave(magnitude_db, xf)
        
        # Plot the frequency response
        plt.semilogx(xf, smoothed_magnitude_db, label=f'Channel {channel + 1}')
    
    plt.title("Frequency Response of the Sweep Signal (1/3 Octave Smoothed)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", ls="--")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlim(20, 20000)  # Set x-axis limit from 20 Hz to 20000 Hz
    plt.ylim(-80, 0)     # Set y-axis limit for better visualization
    plt.legend()
    plt.show()

# Usage
filename = '../measurements/calibration_0521/sweep_pos0.wav'  # Replace with your sweep signal file
channels = [1, 14, 8, 7]  # Channels 2, 15, 9, 8 (0-based index)
plot_frequency_response(filename, channels)
