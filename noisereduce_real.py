import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
from scipy.io import wavfile
import noisereduce as nr
import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.denoise import IterativeWiener
import matplotlib.pyplot as plt

def trim_to_smallest(arr1, arr2):
    min_size = min(len(arr1), len(arr2))
    trimmed_arr1 = arr1[:min_size]
    trimmed_arr2 = arr2[:min_size]
    return trimmed_arr1, trimmed_arr2

"""
Prepare input file
"""
fs, noise_ = wavfile.read("../measurements/ambient_w_arm1.wav")
noise = noise_[fs*2:fs*6:, 0] / np.iinfo(np.int16).max

fs, noisy_signal = wavfile.read("../measurements/calibration/record_pos1.wav")
noisy_signal = (noisy_signal[fs*2:fs*6, 0]) / np.iinfo(np.int16).max

# perform noise reduction
processed_audio = nr.reduce_noise(y=noisy_signal, y_noise=noise, stationary=True, sr=fs, n_std_thresh_stationary=3.0)
wavfile.write("mywav_reduced_noise.wav", fs, processed_audio)
wavfile.write("mywav_noisy_signal.wav", fs, noisy_signal)

noisy_signal_norm = noisy_signal / np.abs(noisy_signal).max()
processed_audio_norm = processed_audio / np.abs(processed_audio).max()

min_val = -80
max_val = -40
plt.figure()
plt.subplot(2, 1, 1)
plt.specgram(noisy_signal_norm, NFFT=256, Fs=fs, vmin=min_val, vmax=max_val)
plt.title("Noisy Signal")
plt.subplot(2, 1, 2)
plt.specgram(
    processed_audio_norm, NFFT=256, Fs=fs, vmin=min_val, vmax=max_val
)
plt.title("Denoised Signal")

plt.tight_layout(pad=0.5)
plt.show()

print('finished')