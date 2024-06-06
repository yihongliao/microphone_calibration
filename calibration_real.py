import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import tf2sos, stft
from scipy.io.wavfile import write
import time
import math
from noise_reduction import noise_reduction
from remove_spikes import remove_spikes3

def crop_signals(signals, start_time, end_time, fs):
    start_sample = np.ceil(start_time * fs)
    end_sample = np.floor(end_time * fs)

    output_signals = signals[start_sample:end_sample,:]
    return output_signals

def extract_measurements(wav_directory, file_names, channels_to_extract):
    # List to store extracted signals
    signals = []

    # Iterate over WAV files and corresponding channels
    for i, file_name in enumerate(file_names):
        # Construct the full path to the WAV file
        wav_file_path = wav_directory + file_name

        # print('read wav file: ', wav_file_path)
        # Read the WAV file
        sample_rate, signal = wavfile.read(wav_file_path)

        if isinstance(signal, list) == True or len(signal.shape) > 1:
            # Extract the specified channel
            print('extract channel: ', channels_to_extract[i])
            channel_data_ = signal[:, channels_to_extract[i]-1]
        else :
            channel_data_ = signal

        if isinstance(channel_data_[0], int) or isinstance(channel_data_[0], np.int16):
            # Scale signal to -1 ~ 1
            channel_data = (channel_data_) / np.iinfo(np.int16).max
        else:
            channel_data = channel_data_

        # plt.plot(channel_data)
        # plt.show()

        # Append the extracted channel to the list
        signals.append(channel_data)

    return sample_rate, signals

def fast_mul(A, B):
    return np.einsum('ij,jk->ik', A, B)

def fast_mul_sum(A, B):
    return np.einsum('ij,jk->k', A, B)

def fast_vec_mul(A, B):
    return np.einsum('j,j->', A, B)

def fast_elem_mul(A, B):
    return np.einsum('ij,ij->ij', A, B)

def AFRC(y, K=1024, remove_spike=True):
    time_start = time.time()
    print('AFRC start, sample size: ', np.shape(y)[1])
    # Parameters
    halfK = math.ceil((K-1)/2) + 1
    I = np.shape(y)[0] # number of microphones
    u = 0.00005

    ITERS = 1

    # initialize
    G = np.ones((I,K),dtype=np.cdouble)
    Gh = np.ones((I,halfK),dtype=np.cdouble)

    # calculate stft of signal
    f, t, Ystft = stft(y, fs=fs, nperseg=K, window=np.ones(K), axis=1, return_onesided=False)
    Ystft = Ystft * K 

    for iter in range(ITERS):
        J = np.zeros(Ystft.shape[2],dtype=np.double)
        C = np.zeros(Ystft.shape[2],dtype=np.double)

        for m in range(Ystft.shape[2]):
            # AFRC Algorithm
            Y = Ystft[:, :halfK, m]
            Zm = fast_elem_mul(Y, Gh)
            
            for i in range(I):
                diag_YiHm = np.diag(Y[i,:].conj().T) # Eq 26
                Dijm = np.zeros((I,halfK),dtype=np.cdouble)
                Zmi = Zm[i,:]
                for j in range(I):
                    Dijm[j,:] = Zmi-Zm[j,:] # Eq 13

                # Eq 25
                dJm_dGiH = fast_mul_sum(Dijm, diag_YiHm)

                # Eq 24
                dC_dGiH = 2*(fast_vec_mul(Gh[i,:],Gh[i,:].conj().T)-1)*Gh[i,:]

                # Eq 31
                lm = np.absolute(fast_vec_mul(dC_dGiH,dJm_dGiH.conj().T)/fast_vec_mul(dC_dGiH,dC_dGiH.conj().T))

                # Eq 27
                dJcm_dGiH = dJm_dGiH + (2*lm*(fast_vec_mul(Gh[i,:],Gh[i,:].conj().T)-1))*Gh[i,:]

                # Eq 21
                Gradient = -u*dJcm_dGiH
                Gh[i,:] = Gh[i,:] + Gradient

                # Compute C cost                
                C[m] = C[m] + pow(np.real(fast_vec_mul(Gh[i,:],Gh[i,:].conj().T))-1, 2)

            # Compute J cost
            Zm = fast_elem_mul(Y, Gh)
            for i in range(I-1):
                for j in range(i+1,I):
                    Dij = Zm[i,:]-Zm[j,:]
                    J[m] = J[m] + np.real(fast_vec_mul(Dij,Dij.conj().T))

            # if m > 1 and np.abs(J[m]-J[m-1]) < 0.001:
            #     break

        print(f'iter: {iter} J:{J[m]} C:{C[m]} m:{m/(Ystft.shape[2]-1)}')

    # plt.plot(J)
    # plt.show()

    # remove noise spike
    if remove_spike:
        Gh = remove_spikes3(Ystft[:,:halfK,:], Gh, 7, 0.05, 1)
        print("noise spike removed")

    G[:,:halfK] = Gh
    if K % 2 == 0:
        G[:,halfK:] = np.conjugate(Gh[:,-2:0:-1])
    else:
        G[:,halfK:] = np.conjugate(Gh[:,-1:0:-1])

    g = np.real(ifft(G,axis=1)) # transform the filter back to time domain
    # g[int(K*0.9):,:] = 0
    time_end = time.time()
    print("AFRC Time: ", time_end-time_start, " s")

    return g, G

def frequency_domain_filter(signal, filter_kernel):
    """
    Filters a signal using a frequency domain filter applied to segments of the signal.
    
    Parameters:
    signal (numpy array): The input time domain signal.
    filter_kernel (numpy array): The frequency domain filter. 
    
    Returns:
    numpy array: The filtered time domain signal.
    """
    filter_length = len(filter_kernel)
    signal_length = len(signal)
    
    # Pad signal with zeros if necessary to make it a multiple of filter_length
    if signal_length % filter_length != 0:
        pad_length = filter_length - (signal_length % filter_length)
        signal = np.pad(signal, (0, pad_length), 'constant')
    
    # Segment the signal into chunks of filter_length
    num_segments = len(signal) // filter_length
    filtered_signal = np.zeros(len(signal))
    
    for i in range(num_segments):
        segment = signal[i*filter_length:(i+1)*filter_length]
        
        # Perform the FFT of the segment
        segment_fft = fft(segment)
        
        # Apply the filter in the frequency domain
        filtered_segment_fft = segment_fft * filter_kernel
        # plt.plot(np.abs(segment_fft))
        # plt.plot(np.abs(filtered_segment_fft))
        # plt.show()
        
        # Perform the inverse FFT to get back to the time domain
        filtered_segment = ifft(filtered_segment_fft)
        
        # Since the result of ifft might have small imaginary parts due to numerical errors, take the real part
        filtered_segment = np.real(filtered_segment)
        
        # Place the filtered segment back into the signal
        filtered_signal[i*filter_length:(i+1)*filter_length] = filtered_segment
    
    # Remove the padding if it was added
    if signal_length % filter_length != 0:
        filtered_signal = filtered_signal[:-pad_length]
    
    return filtered_signal

def normalize_signals(arr):
    """
    Scale a 2D array to the range [-1, 1] using global maximum and minimum values.
    
    Parameters:
    arr (numpy array): The input 2D array.
    
    Returns:
    numpy array: The scaled array.
    """
    # Find the global maximum and minimum values of the entire array
    global_max = np.max(arr)
    global_min = np.min(arr)
    
    # Scale each element to the range [-1, 1]
    scaled_arr = 2 * (arr - global_min) / (global_max - global_min) - 1
    
    return scaled_arr

def calculate_evaluation_metrics(FRS):
    P = len(FRS)
    alphaA = 0
    alphaP = 0
    for i in range(P-1):
        for j in range(i+1, P):
            tmpA = sum(abs(20*np.log10(abs(FRS[i])/abs(FRS[j])))) / K
            tmpP = sum(abs(np.angle(FRS[i])-np.angle(FRS[j]))) / K
            alphaA += tmpA
            alphaP += tmpP
            # print(tmpA, tmpP)
    alphaA = alphaA * 2 / (P*(P-1))
    alphaP = alphaP * 2 / (P*(P-1))
    return alphaA, alphaP

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

def find_starting_index(signal, threshold):
    # Find the index where the signal starts (crosses the threshold)
    starting_index = np.argmax(signal > threshold)
    return starting_index

def align_and_crop_signals2(signals, target_length):
    num_signals = len(signals)
    aligned_signals = []

    signal_lengths = [len(signal) for signal in signals]
    minimum_sig_length = min(signal_lengths)

    # Find the reference signal with the earliest starting point above the threshold
    reference_index = 0
    starting_index = 0

    reference_signal = signals[reference_index][:minimum_sig_length]

    print("signal shift: ", end =" ")
    for i in range(num_signals):
        if i == reference_index:
            aligned_signals.append(reference_signal[starting_index:starting_index+target_length])
            continue

        current_signal = signals[i][:minimum_sig_length]

        # Compute cross-correlation
        cross_corr, shift = gcc_phat(reference_signal, current_signal, max_tau=None, interp=1)
        print(f"{i}={shift}", end =" ")

        # Zero-pad the current signal to align it with the reference signal
        aligned_signal = np.zeros_like(reference_signal)
        if shift > 0:
            aligned_signal[shift:] = current_signal[:-shift]
        else:
            aligned_signal[:shift] = current_signal[-shift:]

        # Crop the aligned signal to the target length
        aligned_signals.append(aligned_signal[starting_index:starting_index+target_length])
    print("")
    aligned_signals = np.array(aligned_signals)

    # plt.figure(2)
    # for i in range(num_signals):
    #     plt.plot(aligned_signals[i])
    # plt.show()

    return aligned_signals

if __name__ == "__main__":

    num_of_mics = 4
    K = 1024
    denoise = False
    remove_spike = False

    wav_directory = '../measurements/calibration_0605_16mics/'
    # List of file names
    calib_file_names = [f'calib_pos{x}.wav' for x in range(num_of_mics)]
    sweep_file_names = [f'sweep_pos{x}.wav' for x in range(num_of_mics)]

    ###############################################################################################################
    # extract calibration signals
    ###############################################################################################################
    # List of channels to extract for each file
    if num_of_mics == 4:
        channels_to_extract = [2, 15, 9, 8]
    elif num_of_mics == 16:
        channels_to_extract = [2, 1, 16, 15, 13, 14, 3, 4, 6, 5, 12, 11, 9, 10, 7, 8]

    fs, noise_w_calib_signals = extract_measurements(wav_directory, calib_file_names, channels_to_extract)
    
    if denoise:
        print("Denoising")
        noise_signal_range = [round(fs*10), round(fs*20)]
        denoised_signals = noise_reduction(noise_w_calib_signals, noise_signal_range, fs, 5)
    else:
        denoised_signals = noise_w_calib_signals

    signal_range = [round(fs*12), round(fs*29)]
    calib_signals = [sig[signal_range[0]:signal_range[1]] for sig in denoised_signals]
    calib_signals = normalize_signals(calib_signals)


    ###############################################################################################################
    # align calibration signals
    ###############################################################################################################
    # align and crop calibration signals
    # Set the target length for cropping
    target_length = round(fs * 20)
    # Align and crop signals
    calib_signals = align_and_crop_signals2(calib_signals, target_length)

    # plot microphone signals
    fig, axs = plt.subplots(len(calib_signals), sharex=True)
    for i, y in enumerate(calib_signals):
        x = np.linspace(0, len(calib_signals[i])/fs, len(calib_signals[i]))
        axs[i].plot(x, y)
        axs[i].set_xlim(0, len(calib_signals[i])/fs)
        axs[i].set_ylim(-1.0, 1.0)
        if i == len(calib_signals) - 1:
            axs[i].set_xlabel('Time [s]')
    plt.show()

    for i, y in enumerate(calib_signals):
        write(wav_directory + f"calib_1ch_pos{i}.wav", fs, y)                                 
    print("Microphone signal files written.")

    ###############################################################################################################
    # calibration
    ###############################################################################################################
    print("Calibrating")
    g, G = AFRC(np.array(calib_signals), K, remove_spike)
    np.savetxt(wav_directory + "G.txt", G.view(float))


    ###############################################################################################################
    # evaluation
    ###############################################################################################################
    print("Filtering")
    G = np.loadtxt(wav_directory + "G.txt").view(complex)

    # fs, sweep_signals = extract_measurements(wav_directory, sweep_file_names, channels_to_extract)
    sweep_signals = calib_signals

    filtered_signals = []
    for i, sweep_signal in enumerate(sweep_signals):
        filtered_signal = frequency_domain_filter(sweep_signal, G[i,:])
        filtered_signals.append(filtered_signal)
        write(wav_directory + f"calib_sweep_pos{i}.wav", fs, filtered_signal)      

    # print(len(fft(sweep_signals, axis=1)))
    # plt.plot(np.abs(fft(sweep_signals, axis=1)[0]))
    # plt.show()
    a, p = calculate_evaluation_metrics(fft(sweep_signals, axis=1))
    a2, p2 = calculate_evaluation_metrics(fft(filtered_signals, axis=1))
    print("before calibration: ",a, p)
    print("After calibration: ", a2, p2)

    cc1, shift1 = gcc_phat(sweep_signals[0], sweep_signals[1], max_tau=None, interp=1)
    cc2, shift2 = gcc_phat(filtered_signals[0], filtered_signals[1], max_tau=None, interp=1)

    print(shift1)
    print(shift2)


