import numpy as np
from wiener2 import wiener_filter

def subtract_noise_from_signal(signal, noise):

    while len(noise) < len(signal):
        noise = np.concatenate((noise, noise[:len(signal) - len(noise)]))
        
    # Calculate FFT of the signal and noise
    signal_fft = np.fft.fft(signal)
    noise_fft = np.fft.fft(noise[:len(signal)])  # Ensure noise has the same length as signal
    
    # Calculate the amplitude spectrum of signal and noise
    signal_amplitude = np.abs(signal_fft)
    noise_amplitude = np.abs(noise_fft)
    
    # Subtract noise amplitude from signal amplitude
    denoised_amplitude = np.maximum(signal_amplitude - noise_amplitude, 0)  # Ensure no negative amplitudes
    
    # Reconstruct the denoised signal with the modified amplitude
    denoised_fft = signal_fft * (denoised_amplitude / (signal_amplitude + 1e-10))  # Add a small constant to avoid division by zero
    denoised_signal = np.fft.ifft(denoised_fft)
    
    return denoised_signal.real  # Take the real part to remove negligible imaginary part


def noise_reduction(mic_signals, signal_range, fs):
    noisy_signals = mic_signals.copy()
    # denoised_signals = wiener_filter(noisy_signals, signal_range, False, 64, 1)  
    denoised_signals = wiener_filter(noisy_signals, signal_range, 15, "full", fs)  

    return denoised_signals