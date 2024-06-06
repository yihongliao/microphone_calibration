import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_frequency_response(wav_file_path, start_freq, end_freq, step_size, step_duration, delay):
    # Read the .wav file
    sample_rate, data = wavfile.read(wav_file_path)
    
    # Normalize data to range of int16
    # data = data / np.iinfo(np.int16).max
    
    freqs = range(start_freq, end_freq + 1, step_size)
    
    # Handle both mono and stereo audio
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    # num_channels = data.shape[1]
    num_channels = 4
    
    # Initialize the plot
    plt.figure(figsize=(12, 6))
    
    for ch in range(num_channels):
        channel_data = data[:, ch]
        # plt.plot(channel_data[:sample_rate*2])
        # plt.show()
        
        # Calculate the number of samples per step
        samples_per_step = int(step_duration * sample_rate)
        samples_delay = round(delay*sample_rate)
        total_steps = len(freqs)
        
        # Initialize lists to hold magnitudes
        magnitudes_db = []
        for i in range(total_steps):
            # Extract the segment for the current step
            segment = channel_data[samples_delay + round((i+0.25) * samples_per_step): samples_delay + round((i + 1 - 0.25) * samples_per_step)]
            # print(freqs[i])
            # print(segment.shape)

            # Compute the FFT of the segment
            fft_segment = np.fft.fft(segment)
            fft_magnitude = np.abs(fft_segment) * 2 / len(segment)
            # plt.plot(np.linspace(0,sample_rate,len(segment)), fft_magnitude)
            # plt.show()
            
            # Find the maximum value of the FFT magnitude for this segment
            max_magnitude = np.max(fft_magnitude[:len(segment) // 2])  # Use only the first half of the FFT result

            # Convert the magnitude to dB
            if max_magnitude > 0:
                max_magnitude_db = 20 * np.log10(max_magnitude)
            else:
                max_magnitude_db = -np.inf  # To handle log(0)
            
            # Append the max magnitude in dB to the list
            magnitudes_db.append(max_magnitude_db)
        
        # Plot the frequency response for this channel
        plt.plot(freqs, magnitudes_db, marker='o', label=f'Channel {ch+1}')
    
    plt.xlim((start_freq, end_freq))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Response')
    plt.grid(True)
    plt.legend()
    plt.show()

# Usage
wav_file_path = '../measurements/test.wav'  # Replace with your file path
start_frequency = 0    # Hz
end_frequency = 20000  # Hz
frequency_step = 500   # Hz
step_duration = 1    # seconds
delay = 0.28

plot_frequency_response(wav_file_path, start_frequency, end_frequency, frequency_step, step_duration, delay)
