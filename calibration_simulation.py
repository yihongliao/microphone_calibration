"""
This example creates a room with reverberation time specified by inverting Sabine's formula.
This results in a reverberation time slightly longer than desired.
The simulation is pure image source method.
The audio sample with the reverb added is saved back to `examples/samples/guitar_16k_reverb.wav`.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.signal import tf2sos, stft
from scipy.fft import ifft, fft, fftshift
import scipy.io.wavfile
import scipy.io
import random
import math
from statistics import mean 
import csv
import os
import time
import pyroomacoustics as pra
from noise_reduction import noise_reduction
from remove_spikes import remove_spikes3


methods = ["ism", "hybrid", "anechoic"]

def initialize_csv(filename, snr_values, t60_values):
    if os.path.exists(filename):
        print(filename, "exist!")
    else:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write the header row with T60 values
            writer.writerow(['SNR'] + t60_values)
            
            # Initialize the table with placeholder values
            for snr in snr_values:
                writer.writerow([snr] + [''] * len(t60_values))

def fill_value(filename, snr, t60, value):
    with open(filename, 'r', newline='\n') as csvfile:
        rows = list(csv.reader(csvfile))
        
    # Find the row index corresponding to the SNR value
    snr_index = [row[0] for row in rows].index(str(snr))
    
    # Find the column index corresponding to the T60 value
    t60_index = rows[0].index(str(t60))
    
    # Update the value at the specified cell
    rows[snr_index][t60_index] = value
    
    # Write the updated rows back to the CSV file
    with open(filename, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

def pink_psd(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

def pink_noise(N):
        X_white = np.fft.rfft(np.random.randn(N))
        S = pink_psd(np.fft.rfftfreq(N))
        # Normalize S
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S
        return np.fft.irfft(X_shaped)

def arm_noise(N):
    fs, data_ = wavfile.read("../measurements/ambient_w_arm1.wav")
    # data = np.resize(data_[:, 0], N)
    data = data_[:, 0]
    data = np.pad(data, (0, N-len(data)))

    spectrum = np.abs(fft(data))

    noise = np.random.normal(0, 1, len(spectrum))
    noise_fft = fft(noise)
    noise_fft_with_amplitude_spectrum = noise_fft * spectrum / np.abs(noise_fft)
    noise_with_amplitude_spectrum = np.real(np.fft.ifft(noise_fft_with_amplitude_spectrum))
    return noise_with_amplitude_spectrum

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

        if isinstance(signal[0], list) == True:
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
        aligned_signal[max(0, shift):min(len(current_signal) + shift, len(current_signal))] = current_signal[max(-shift, 0):min(len(current_signal), len(current_signal)-shift)]

        # Crop the aligned signal to the target length
        aligned_signals.append(aligned_signal[starting_index:starting_index+target_length])
    print("")
    aligned_signals = np.array(aligned_signals)

    # plt.figure(2)
    # for i in range(num_signals):
    #     plt.plot(aligned_signals[i])
    # plt.show()

    return aligned_signals

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
    u = 1.0 # step size

    ITERS = 1

    # initialize
    G = np.ones((I,K),dtype=np.cdouble)
    Gh = np.ones((I,halfK),dtype=np.cdouble)

    # calculate stft of signal
    f, t, Ystft = stft(y, fs=fs, nperseg=K, window=np.ones(K), axis=1, return_onesided=False)
    Ystft = Ystft * K

    J = np.zeros(Ystft.shape[2],dtype=np.double)
    C = np.zeros(Ystft.shape[2],dtype=np.double)

    for iter in range(ITERS):
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

            if m > 1 and np.abs(J[m]-J[m-1]) < 0.00001:
                break

        print(f'iter: {iter} J:{J[m]} C:{C[m]}')

    # remove noise spike
    if remove_spike:
        Gh = remove_spikes3(Ystft[:,:halfK,:], Gh, 7, 0.1, 1)
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

def generate_mic_array_movements(rows, cols, d, p):
    coordinates = []

    for i in range(rows):
        if i % 2 == 0:
            for j in range(cols):
                ri = random.uniform(-1, 1) * p
                rj = random.uniform(-1, 1) * p
                coordinates.append((i*d+ri, 0, j*d+rj))
        else:
            for j in range(cols - 1, -1, -1):
                ri = random.uniform(-1, 1) * p
                rj = random.uniform(-1, 1) * p
                coordinates.append((i*d+ri, 0, j*d+rj))

    return np.array(coordinates).T

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a dry sound sample. Saves it into `./examples/samples`."
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=methods,
        default=methods[0],
        help="Simulation method to use",
    )
    parser.add_argument('-c', '--calibration',
                    action='store_true')  # on/off flag
    parser.add_argument('-e', '--evaluate',
                    action='store_true')  # on/off flag
    parser.add_argument('-p', '--plot',
                    action='store_true')  # on/off flag
    parser.add_argument('-w', '--write',
                    action='store_true')  # on/off flag
    args = parser.parse_args()

    ###############################################################################################################
    # Parameters
    ###############################################################################################################
    fs = 44100
    K = 1024
    num_of_mics = 4

    # The desired reverberation time and dimensions of the room
    SNRs = [15.0, 20.0, 25.0, 35.0]  # signal-to-noise ratio in dB
    rt60_tgts = [0.212, 0.28, 0.31, 0.39, 0.44]  # seconds
    trials = 1
    room_dim = [7.1, 6.0, 3.0]  # meters
    d = 0.126
    precision = 0.0005
    signal_range = [fs*10, fs*20]

    plot_figure = args.plot
    calibration = args.calibration
    evaluate = args.evaluate
    write_eval_result = args.write
    add_noise = True
    denoise = True
    remove_spike = True

    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    # fs, audio = wavfile.read("samples/guitar_16k.wav")
    fs, audio = wavfile.read("../signal/white_noise_w_blank.wav")
    audio = audio / np.iinfo(np.int16).max

    path = f"simulation_calibrations/p_{precision}"
    if not os.path.exists(path):
        os.mkdir(f"simulation_calibrations/p_{precision}") 

    if write_eval_result:
        filename_A = f"simulation_calibrations/p_{precision}/AlphaA.csv"
        filename_P = f"simulation_calibrations/p_{precision}/AlphaP.csv"
        # Initialize the CSV file with placeholders
        initialize_csv(filename_A, SNRs, rt60_tgts)
        initialize_csv(filename_P, SNRs, rt60_tgts)

    ###############################################################################################################
    # simulate microphone ICS
    ###############################################################################################################
    ICS = []
    ICS.append(signal.iirfilter(17, [400, 21650], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=fs,
                       output='sos'))
    ICS.append(signal.iirfilter(17, [500, 21550], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=fs,
                       output='sos'))
    ICS.append(signal.iirfilter(17, [600, 21450], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=fs,
                       output='sos'))
    ICS.append(signal.iirfilter(17, [700, 21350], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=fs,
                       output='sos'))
    ICS[1][0,:3] = ICS[1][0,:3] * pow(10,-1/20)
    ICS[2][0,:3] = ICS[2][0,:3] * pow(10,3/20)
    ICS[3][0,:3] = ICS[3][0,:3] * pow(10,-7/20)

    ICSFR_O = []
    u = signal.unit_impulse(K)
    for i in range(len(ICS)):
        ics_imp = signal.sosfilt(ICS[i], u)
        Y = fft(ics_imp)
        ICSFR_O.append(Y)

    # print("Original")
    # calculate_evaluation_metrics(ICSFR_O)

    # plot original ICS frequency response
    if plot_figure:
        fig, ax = plt.subplots(2, figsize=(5, 5), sharex=True)
        w = np.linspace(0, fs*(K-1)/K, K)
        for i, icsfr_o in enumerate(ICSFR_O):
            Yabs = abs(icsfr_o)
            ax[0].plot(w/1000, 20 * np.log10(np.maximum(Yabs, 1e-5)), label=f"CH {i}")
            ax[1].plot(w/1000, np.angle(icsfr_o))
            ax[0].set_ylabel('Amplitude [dB]')
            ax[1].set_xlabel('Frequency [kHz]')
            ax[1].set_ylabel('Phase [rad]')
            ax[0].axis((0.01, fs/2000, -100, 10))
            ax[1].axis((0.01, fs/2000, -5, 5))
            ax[0].grid(which='both', axis='both')
            ax[1].grid(which='both', axis='both')   
        ax[0].legend()
        fig.suptitle('Original microphone ICS Frequency Response')

    # plt.show()
        
    ###############################################################################################################
    # Start simulation
    ###############################################################################################################
    for SNR in SNRs:
        for rt60_tgt in rt60_tgts:
            alphaAs = []
            alphaPs = []
            for T in range(trials):
                if calibration:
                    print("p: ", precision, " SNR: ", SNR, " RT60: ", rt60_tgt, " Trial: ", T)

                    clean_signals = []
                    array_signals = []
                    denoised_signals = []
                    calib_signals = []
                    
                    # We invert Sabine's formula to obtain the parameters for the ISM simulator
                    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

                    mic_array_movements = generate_mic_array_movements(2, 2, d, precision)
                    print(mic_array_movements)
                    mic_array_initial_loc = np.c_[
                            [2.45, 2.8, 1.3], [2.45, 2.8, 1.3-d], [2.45-d, 2.8, 1.3-d], [2.45-d, 2.8, 1.3]
                        ]

                    if add_noise:
                        print("Noise added, Denoise: ", denoise)

                    for k in range(num_of_mics):
                        print(f"processing mic: {k}")
                        ###############################################################################################################
                        # Create the room
                        ###############################################################################################################
                        if args.method == "ism":
                            room = pra.ShoeBox(
                                room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
                            )
                        elif args.method == "hybrid":
                            room = pra.ShoeBox(
                                room_dim,
                                fs=fs,
                                materials=pra.Material(e_absorption),
                                max_order=3,
                                ray_tracing=True,
                                air_absorption=True,
                            )
                        elif args.method == "anechoic":
                            room = pra.AnechoicRoom(3, fs=fs)

                        # place the source in the room
                        room.add_source([2.45, 2.3, 1.3], signal=audio)

                        ###############################################################################################################
                        # Add microphones
                        ###############################################################################################################
                        # define the locations of the microphones
                        move = mic_array_movements[:,k]
                        mic_locs = mic_array_initial_loc + move[:,np.newaxis]
                        # print(mic_locs)
                        # finally place the array in the room
                        room.add_microphone_array(mic_locs)

                        ###############################################################################################################
                        # Run the simulations (this will also build the RIR automatically)
                        ###############################################################################################################
                        room.simulate()
                        S = room.mic_array.signals
                        clean_signals.append(S/100)

                        signals = []
                        noise = arm_noise(len(S[0]))
                        for i in range(num_of_mics):
                            s = S[i,:] / 100
                            mic_signal = s

                            if add_noise:
                                # noise = pink_noise(len(s))
                                # noise = arm_noise(len(s))
                                Es = sum(np.power(s[signal_range[0]:signal_range[1]], 2))
                                En = sum(np.power(noise[signal_range[0]:signal_range[1]], 2))
                                alpha = np.sqrt(Es/(10**(SNR/10)*En))
                                mic_signal = s + alpha*noise
                            
                            signals.append(mic_signal)

                        ###############################################################################################################
                        # Create microphone filtered signals
                        ###############################################################################################################
                        mic_signals = []                       
                        for i in range(len(signals)):
                            mic_signal = signal.sosfilt(ICS[i], signals[i])
                            mic_signals.append(mic_signal)

                        array_signals.append(mic_signals)  
            
                    if denoise:
                        print("denoising...")
                        ##################
                        noise_w_calib_signals = [sigs[pos] for pos, sigs in enumerate(array_signals)]
                        for i, y in enumerate(noise_w_calib_signals):
                            scipy.io.wavfile.write(f"simulation_calibrations/noisy_signal{i}_{SNR}_{rt60_tgt}.wav", fs, y)             
                        denoised_signals = noise_reduction(noise_w_calib_signals, signal_range, fs)
                        calib_signals = [sig[signal_range[0]:signal_range[1]] for sig in denoised_signals]
                    else:
                        calib_signals = [sigs[pos][signal_range[0]:signal_range[1]] for pos, sigs in enumerate(array_signals)]

                    # save calibration signals             
                    for i, y in enumerate(calib_signals):
                        scipy.io.wavfile.write(f"simulation_calibrations/calibration_signal{i}_{SNR}_{rt60_tgt}.wav", fs, y)                                 
                    print("Microphone signal files written.")
                        
                    if plot_figure:
                        min_val = -140
                        max_val = -70
                        plt.figure()
                        plt.subplot(3, 1, 1)
                        plt.specgram(clean_signals[0][0], NFFT=1024, noverlap=512, Fs=fs, vmin=min_val, vmax=max_val)
                        plt.title("Original Signal")
                        plt.subplot(3, 1, 2)
                        plt.specgram(array_signals[0][0], NFFT=1024, noverlap=512, Fs=fs, vmin=min_val, vmax=max_val)
                        plt.title("Noisy Mic Signal")
                        if denoise:
                            plt.subplot(3, 1, 3)
                            plt.specgram(denoised_signals[0], NFFT=1024, noverlap=512, Fs=fs, vmin=min_val, vmax=max_val)
                            plt.title("Denoise Mic Signal")


                    # plot microphone signals
                    if plot_figure:
                        fig, axs = plt.subplots(len(calib_signals), sharex=True)
                        for i, y in enumerate(calib_signals):
                            x = np.linspace(0, len(calib_signals[i])/fs, len(calib_signals[i]))
                            axs[i].plot(x, y)
                            axs[i].set_xlim(0, len(calib_signals[i])/fs)
                            axs[i].set_ylim(-0.05, 0.05)
                            if i == len(calib_signals) - 1:
                                axs[i].set_xlabel('Time [s]')

                    if args.method == "ism":
                        # measure the reverberation time
                        rt60 = room.measure_rt60()
                        # print("The desired RT60 was {}".format(rt60_tgt))
                        print("The measured RT60 is {}".format(rt60[0, 0]))

                        if plot_figure:
                            # plot the RIRs
                            select = None  # plot all RIR
                            # # select = (2, 0)  # uncomment to only plot the RIR from mic 2 -> src 0
                            # # select = [(0, 0), (2, 0)]  # only mic 0 -> src 0, mic 2 -> src 0
                            fig, axes = room.plot_rir(select=select, kind="ir")  # impulse responses
                            # fig, axes = room.plot_rir(select=select, kind="tf")  # transfer function
                            fig, axes = room.plot_rir(select=select, kind="spec")  # spectrograms

                   

                    ###############################################################################################################
                    # Calibration
                    ###############################################################################################################
                    # extract measurement signals
                    # Specify the directory containing your WAV files
                    wav_directory = ''
                    # List of file names
                    # file_names = [f'record_pos{x}.wav' for x in [1, 4, 2, 3]]
                    file_names = [f"simulation_calibrations/calibration_signal{i}_{SNR}_{rt60_tgt}.wav" for i in range(4)]

                    # List of channels to extract for each file
                    # channels_to_extract = [2, 15, 9, 8]
                    channels_to_extract = []
                    fs, signals = extract_measurements(wav_directory, file_names, channels_to_extract)

                    # align and crop calibration signals
                    # Set a threshold for signal starting point
                    threshold = 0.1
                    # Set the target length for cropping
                    target_length = round(fs * 10)
                    # Align and crop signals
                    aligned_and_cropped_signals = align_and_crop_signals(signals, target_length, threshold=threshold)
                    
                    # If already knew the delay
                    # aligned_and_cropped_signals = np.array([signal for signal in signals])
                    # print(aligned_and_cropped_signals[0][:10])

                    # perform AFRC
                    g, G = AFRC(aligned_and_cropped_signals, K, remove_spike)

                    # np.savetxt(f"simulation_calibrations/G_{SNR}_{rt60_tgt}.txt", g)
                    np.savetxt(f"simulation_calibrations/p_{precision}/G_{SNR}_{rt60_tgt}_{T}.txt", G.view(float))
                    # np.savetxt(f"simulation_calibrations/p_{precision}/g_{SNR}_{rt60_tgt}_{T}_2.txt", g.view(float))


                if evaluate:
                    ###############################################################################################################
                    # Load calibration filter
                    ###############################################################################################################
                    G = np.loadtxt(f"simulation_calibrations/p_{precision}/G_{SNR}_{rt60_tgt}_{T}.txt").view(complex)
                    # plt.show()
                    # load calibration filter from Matlab
                    # g = scipy.io.loadmat(f"simulation_calibrations/p_{precision}/g_{SNR}_{rt60_tgt}_{T}_2.mat")
                    # G = fft(g['g2'], axis=1)

                    ###############################################################################################################
                    # Calculate the microphone frequency response after calibration
                    ###############################################################################################################
                    ICSFR_C = []
                    u = signal.unit_impulse(K)
                    for i in range(len(ICS)):
                        ics_imp = signal.sosfilt(ICS[i], u)
                        Y = fft(ics_imp)*G[i,:]
                        ICSFR_C.append(Y)

                    if plot_figure:
                        # plot the microphone frequency response after calibration
                        fig, ax = plt.subplots(2, figsize=(5, 5), sharex=True)
                        w = np.linspace(0, fs*(K-1)/K, K)
                        for i, icsfr_c in enumerate(ICSFR_C):
                            Yabs = abs(icsfr_c)
                            ax[0].plot(w/1000, 20 * np.log10(np.maximum(Yabs, 1e-5)), label=f"CH {i}")
                            ax[1].plot(w/1000, np.angle(icsfr_c))
                            ax[0].set_ylabel('Amplitude [dB]')
                            ax[1].set_xlabel('Frequency [kHz]')
                            ax[1].set_ylabel('Phase [rad]')
                            ax[0].axis((0.01, fs/2000, -100, 10))
                            ax[1].axis((0.01, fs/2000, -5, 5))
                            ax[0].grid(which='both', axis='both')
                            ax[1].grid(which='both', axis='both')   
                        ax[0].legend()
                        fig.suptitle('Calibrated microphone ICS Frequency Response')

                    ###############################################################################################################
                    # calculate evaluation criteria
                    ###############################################################################################################
                    alphaA, alphaP = calculate_evaluation_metrics(ICSFR_C)
                    alphaAs.append(alphaA)
                    alphaPs.append(alphaP)
                    print("calibrated SNR: ", SNR, " RT60: ", rt60_tgt, " p: ", precision, " Trial: ", T ," AlphaA: ", alphaA, " AlphaP: ", alphaP)
                    print("\n")


                if plot_figure:
                    plt.tight_layout()
                    plt.show()

            alphaA_mean = mean(alphaAs)
            alphaP_mean = mean(alphaPs)
            
            if write_eval_result:
                fill_value(filename_A, SNR, rt60_tgt, alphaA_mean)
                fill_value(filename_P, SNR, rt60_tgt, alphaP_mean)
                print("written to csv")

            print("==============================================================================================================")
            print("calibrated SNR: ", SNR, " RT60: ", rt60_tgt," d: ", d, trials, " Trials " ," AlphaA: ", alphaA_mean, " AlphaP: ", alphaP_mean)
            print("==============================================================================================================")
            print("\n")