from wiener import wiener_filter
import mynoisereduce as mnr
import numpy as np
import matplotlib.pyplot as plt

def noise_reduction(mic_signals, signal_range):
    noisy_signals = mic_signals.copy()

    # spectral subtraction
    # for i, noisy_signal in enumerate(noisy_signals):
    #     noise = noisy_signal[0:signal_range[0]]
    #     noisy_signals[i] = mnr.reduce_noise(y=noisy_signal, y_noise=noise, stationary=True, sr=44100, freq_mask_smooth_hz=100, n_std_thresh_stationary=-0.5)
    
    # Wiener filter
    nsysignals = np.array(noisy_signals).T
    denoise_signals = wiener_filter(nsysignals, signal_range, 32, 1)
    denoised_signals = [np.array(sig) for sig in list(denoise_signals.T)]

    return denoised_signals