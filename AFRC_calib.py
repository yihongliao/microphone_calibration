import numpy as np
from scipy.fft import ifft, fft
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.io import savemat
import time

def find_starting_index(signal, threshold):
    # Find the index where the signal starts (crosses the threshold)
    starting_index = np.argmax(signal > threshold)
    return starting_index

def extract_measurements(wav_directory, file_names, channels_to_extract):
    # List to store extracted signals
    signals = []

    # Iterate over WAV files and corresponding channels
    for i, file_name in enumerate(file_names):
        # Construct the full path to the WAV file
        wav_file_path = wav_directory + file_name

        print('read wav file: ', wav_file_path)
        # Read the WAV file
        sample_rate, signal = wavfile.read(wav_file_path)

        if isinstance(signal[0], list) == True:
            # Extract the specified channel
            print('extract channel: ', channels_to_extract[i])
            channel_data_ = signal[:, channels_to_extract[i]-1]
        else :
            channel_data_ = signal

        if isinstance(channel_data_[0], int):
            # Scale signal to -1 ~ 1
            channel_data = (channel_data_) / np.iinfo(np.int16).max
        else:
            channel_data = channel_data_

        # plt.plot(channel_data)
        # plt.show()

        # Append the extracted channel to the list
        signals.append(channel_data)

    return sample_rate, signals

def gcc_phat(sig, refsig, max_tau=None, interp=1):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift
    
    return cc, shift

def align_and_crop_signals(signals, target_length, threshold=0.003):
    num_signals = len(signals)
    aligned_signals = []

    signal_lengths = [len(signal) for signal in signals]
    minimum_sig_length = min(signal_lengths)

    # Find the reference signal with the earliest starting point above the threshold
    starting_indices = [find_starting_index(signal, threshold) for signal in signals]
    reference_index = np.argmin(starting_indices)
    starting_index = starting_indices[reference_index] + round(0.1*target_length)

    reference_signal = signals[reference_index][:minimum_sig_length]

    for i in range(num_signals):
        if i == reference_index:
            aligned_signals.append(reference_signal[starting_index:starting_index+target_length])
            continue

        current_signal = signals[i][:minimum_sig_length]

        # Compute cross-correlation
        cross_corr, shift = gcc_phat(reference_signal, current_signal, max_tau=None, interp=1)
        print(shift)

        # Zero-pad the current signal to align it with the reference signal
        aligned_signal = np.zeros_like(reference_signal)
        aligned_signal[max(0, shift):min(len(current_signal) + shift, len(current_signal))] = current_signal[max(-shift, 0):min(len(current_signal), len(current_signal)-shift)]

        # Crop the aligned signal to the target length
        aligned_signals.append(aligned_signal[starting_index:starting_index+target_length])

    aligned_signals = np.array(aligned_signals)

    plt.figure(2)
    for i in range(num_signals):
        plt.plot(aligned_signals[i])
    plt.show()

    return aligned_signals

def prt10(v):
    for i in range(10):
        print(v[i])
    print('finish\n')

def prt10N(v):
    N = v.shape[0]
    for n in range(N):
        for i in range(10): 
            print(v[n][i])
        print('\n')
    print('finish\n')

def fast_mul(A, B):
    return np.einsum('ij,jk->ik', A, B)

def fast_mul_sum(A, B):
    return np.einsum('ij,jk->k', A, B)

def fast_vec_mul(A, B):
    return np.einsum('j,j->', A, B)

def fast_elem_mul(A, B):
    return np.einsum('ij,ij->ij', A, B)

def AFRC(y):
    print('AFRC start')
    # Parameters
    K = 2048 # number of frequency bins
    I = np.shape(y)[0] # number of microphones
    print('sample size: ', np.shape(y)[1])
    u = 0.5 # step size

    N = K # number of sample points in a frame
    O = int(N/2) # overlap length
    ITERS = 1

    # initialize
    G = np.ones((I,K),dtype=np.cdouble)
    G = G/np.sqrt(K)

    J_iter = []
    C_iter = []
    T_iter = []
    for iter in range(ITERS):
        m = 1
        J = []
        C = []
        T = []
        while (N-O)*(m-1)+N < np.shape(y)[1]:
            # AFRC Algorithm
            frame_start = (N-O)*(m-1)
            frame_end = (N-O)*(m-1)+N
            Y = fft(y[:,frame_start:frame_end],axis=1)
            Zm = fast_elem_mul(Y, G)
            
            J.append(0)
            C.append(0)
            T.append(0)
            for i in range(I):
                diag_YiHm = np.diag(Y[i,:].conj().T) # Eq 26
                Dijm = np.zeros((I,K),dtype=np.cdouble)
                Zmi = Zm[i,:]
                for j in range(I):
                    Dijm[j,:] = Zmi-Zm[j,:] # Eq 13

                # Eq 25
                dJm_dGiH = fast_mul_sum(Dijm, diag_YiHm)

                # Eq 24
                dC_dGiH = 2*(0.0000001 + fast_vec_mul(G[i,:],G[i,:].conj().T)-1)*G[i,:]

                # Eq 31
                lm = np.absolute(fast_vec_mul(dC_dGiH,dJm_dGiH.conj().T)/fast_vec_mul(dC_dGiH,dC_dGiH.conj().T))

                # Eq 27
                dJcm_dGiH = dJm_dGiH + (2*lm*(fast_vec_mul(G[i,:],G[i,:].conj().T)-1))*G[i,:]

                # Eq 21
                Gradient = -u*dJcm_dGiH
                G[i,:] = G[i,:] + Gradient

                # Compute cost
                for j in range(I):
                    J[-1] = J[-1] + fast_vec_mul(Dijm[j,:],Dijm[j,:].conj().T)
                C[-1] = C[-1] + pow(fast_vec_mul(G[i,:],G[i,:].conj().T)-1, 2)
                # print(C[-1])
                T[-1] = J[-1] + lm * C[-1]

            m = m + 1

        J_iter.append(sum(J))
        C_iter.append(sum(C))
        T_iter.append(sum(T))
        print(f'iter: {iter} J:{J_iter[-1]} C:{C_iter[-1]} T: {T_iter[-1]}')
        
    G = G*np.sqrt(K)
    g = np.real(ifft(G,axis=1)) # transform the filter back to time domain
    # g[int(K*0.9):,:] = 0

    return g, G

# Example usage
if __name__ == "__main__":

    SNR = 35.0
    rt60 = 0.4

    # extract measurement signals
    # Specify the directory containing your WAV files
    wav_directory = ''
    
    # List of file names
    # file_names = [f'record_pos{x}.wav' for x in [1, 4, 2, 3]]
    file_names = [f"simulation_calibrations/calibration_signal{i}_{SNR}_{rt60}.wav" for i in range(4)]
    print(file_names)

    # List of channels to extract for each file
    # channels_to_extract = [2, 15, 9, 8]
    channels_to_extract = []
    fs, signals = extract_measurements(wav_directory, file_names, channels_to_extract)

    # # align and crop calibration signals
    # # Set a threshold for signal starting point
    # threshold = 0.1
    # # Set the target length for cropping
    # target_length = round(fs * 9.8)
    # # Align and crop signals
    # aligned_and_cropped_signals = align_and_crop_signals(signals, target_length, threshold=threshold)
    
    # If already knew the delay
    aligned_and_cropped_signals = np.array([signal for signal in signals])

    # perform AFRC
    g, G = AFRC(aligned_and_cropped_signals)

    # np.savetxt(f"simulation_calibrations/G_{SNR}_{rt60}.txt", g)
    np.savetxt(f"simulation_calibrations/G_{SNR}_{rt60}.txt", G.view(float))

    print('filter saved')

