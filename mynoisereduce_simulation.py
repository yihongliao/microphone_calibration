import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
from scipy.io import wavfile
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'mynoisereduce'))
import noisereduce.noisereduce as nr

# import noisereduce as nr
import numpy as np

import matplotlib.pyplot as plt

def trim_to_smallest(arr1, arr2):
    min_size = min(len(arr1), len(arr2))
    trimmed_arr1 = arr1[:min_size]
    trimmed_arr2 = arr2[:min_size]
    return trimmed_arr1, trimmed_arr2

snr = 20 #dB

"""
Prepare input file
"""
fs, noise_ = wavfile.read("../measurements/ambient_w_arm1.wav")
noise = noise_[fs*2:fs*6, 0] 
noise = noise / np.abs(noise).max()

fs, signal = wavfile.read("../signal/Chirp_Sound_cycle.wav")
signal = (signal[fs*2:fs*6]) / np.iinfo(np.int16).max

Es = sum(np.power(signal, 2))
En = sum(np.power(noise, 2))
alpha = np.sqrt(Es/(10**(snr/10)*En))
noisy_signal = signal+alpha*noise

# perform noise reduction
processed_audio = nr.reduce_noise(y=noisy_signal, y_noise=noise, stationary=True, sr=fs, n_std_thresh_stationary=1.0)
# wavfile.write("mywav_reduced_noise.wav", fs, processed_audio)
# wavfile.write("mywav_clean_signal.wav", fs, signal)
# wavfile.write("mywav_noise_signal.wav", fs, alpha*noise)
# wavfile.write("mywav_noisy_signal.wav", fs, noisy_signal)

noisy_signal_norm = noisy_signal / np.abs(noisy_signal).max()
signal_norm = signal / np.abs(signal).max()
processed_audio_norm = processed_audio / np.abs(processed_audio).max()

min_val = -80
max_val = -40
plt.figure()
plt.subplot(3, 1, 1)
plt.specgram(signal_norm, NFFT=256, Fs=fs, vmin=min_val, vmax=max_val)
plt.title("Original Signal")
plt.subplot(3, 1, 2)
plt.specgram(noisy_signal_norm, NFFT=256, Fs=fs, vmin=min_val, vmax=max_val)
plt.title("Noisy Signal")
plt.subplot(3, 1, 3)
plt.specgram(
    processed_audio_norm, NFFT=256, Fs=fs, vmin=min_val, vmax=max_val
)
plt.title("Denoised Signal")

plt.tight_layout(pad=0.5)
plt.show()

print('finished')