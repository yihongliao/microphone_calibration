import numpy as np

def cinterp(complex_vector):
    
    missing_indices = np.nonzero(np.isnan(complex_vector))[0]

    # Interpolate missing values
    for idx in missing_indices:
        if not np.isnan(complex_vector[idx]):
            continue
        
        # Find nearest non-missing values
        prev_idx = np.argwhere(~np.isnan(complex_vector[:idx]))[-1][0]
        next_idx = np.argwhere(~np.isnan(complex_vector[idx+1:]))[0][0] + idx + 1
        
        # Interpolate magnitude
        # Compute common ratio for geometric series interpolation
        mag_prev = abs(complex_vector[prev_idx])
        mag_next = abs(complex_vector[next_idx])
        common_ratio = (mag_next / mag_prev) ** (1 / (next_idx - prev_idx))
        
        # Interpolate magnitude using geometric series
        mag_interp = mag_prev * common_ratio ** np.arange(next_idx - prev_idx + 1)
        
        # Wrap phases to -pi to pi range
        phase_prev = np.angle(complex_vector[prev_idx])
        phase_next = np.angle(complex_vector[next_idx])
        phase_diff = np.angle(np.exp(1j*(phase_next - phase_prev)))
        phase_diff_wrapped = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
        
        # Interpolate phase taking into account wrapping
        phase_interp = np.linspace(phase_prev, phase_prev + phase_diff_wrapped, next_idx - prev_idx + 1)
        
        # Convert interpolated magnitude and phase to complex numbers
        interpolated_complex = mag_interp * np.exp(1j * phase_interp)
        
        # Fill in missing values
        complex_vector[prev_idx+1:next_idx] = interpolated_complex[1:-1]
    return complex_vector


def find_spikes(YSEG, threshold=0.1, spike_size=3, K=1024):
    num_of_mics = YSEG.shape[0]
    # find gradient
    absGRAD_mean = np.zeros((K, 1))
    phGRAD_mean = np.zeros((K, 1))

    for i in range(num_of_mics - 1):
        for j in range(i + 1, num_of_mics):
            YnabsGRAD = np.gradient(np.abs(YSEG[i, :, :]) / np.abs(YSEG[j, :, :]), axis=1)
            YnphGRAD = np.gradient(np.angle(np.exp(1j * (np.angle(YSEG[i, :, :]) - np.angle(YSEG[j, :, :])))), axis=1)
            absGRAD_mean += np.mean(np.abs(YnabsGRAD), axis=1, keepdims=True)
            phGRAD_mean += np.mean(np.abs(YnphGRAD), axis=1, keepdims=True)

    absGRAD_mean /= (num_of_mics * (num_of_mics - 1) / 2)
    phGRAD_mean /= (num_of_mics * (num_of_mics - 1) / 2)

    bad_freqs_idxs = np.where(phGRAD_mean > threshold)[0]
    spike_idxs = []

    for bfi in bad_freqs_idxs:
        spikes = range(bfi - spike_size, bfi + spike_size + 1)
        for s in spikes:
            if s not in spike_idxs and 0 < s < K-1:
                spike_idxs.append(s)
    
    return spike_idxs

def remove_spikes(YSEG, threshold=0.5, spike_size=3, K=1024):
    spike_idxs = find_spikes(YSEG, threshold, spike_size, K)
    YSEG[:,spike_idxs,:] = np.nan
    for i in range(YSEG.shape[0]):
        for j in range(YSEG.shape[2]):
            YSEG[i,:,j] = cinterp(YSEG[i,:,j])
    return YSEG
