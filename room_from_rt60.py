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
from scipy.signal import tf2sos
from scipy.fft import ifft, fft, fftshift
import scipy.io.wavfile
import scipy.io
import noisereduce as nr
import pyroomacoustics as pra

methods = ["ism", "hybrid", "anechoic"]

def pink_psd(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

def pink_noise(N):
        X_white = np.fft.rfft(np.random.randn(N))
        S = pink_psd(np.fft.rfftfreq(N))
        # Normalize S
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S
        return np.fft.irfft(X_shaped)

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
    alphaA = alphaA / (P*(P-1))
    alphaP = alphaP / (P*(P-1))
    print("AlphaA: ", alphaA)
    print("AlphaP: ", alphaP)

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
    args = parser.parse_args()

    ###############################################################################################################
    # Parameters
    ###############################################################################################################
    fs = 44100
    K = 1024
    num_of_mics = 4

    # The desired reverberation time and dimensions of the room
    SNR = 35.0  # signal-to-noise ratio in dB
    rt60_tgt = 0.4  # seconds
    room_dim = [7.1, 6.0, 3.0]  # meters
    signal_range = [fs*2, fs*12]

    plot_figure = args.plot
    calibration = args.calibration
    evaluate = args.evaluate
    add_noise = True
    denoise = False
    n_std_thresh_stationary = 0.1

    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    # fs, audio = wavfile.read("samples/guitar_16k.wav")
    fs, audio = wavfile.read("../signal/white_noise_15s.wav")
    audio = audio / np.iinfo(np.int16).max

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

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

    print("Original")
    calculate_evaluation_metrics(ICSFR_O)

    # plot original ICS frequency response
    if plot_figure:
        fig, ax = plt.subplots(2, sharex=True)
        w = np.linspace(0, fs*(K-1)/K, K)
        for icsfr_o in ICSFR_O:
            Yabs = abs(icsfr_o)
            ax[0].plot(w, 20 * np.log10(np.maximum(Yabs, 1e-5)))
            ax[1].plot(w, np.angle(icsfr_o))
            ax[0].set_ylabel('Amplitude [dB]')
            ax[1].set_xlabel('Frequency [Hz]')
            ax[1].set_ylabel('Phase [rad]')
            ax[0].axis((10, fs/2, -100, 10))
            ax[1].axis((10, fs/2, -5, 5))
            ax[0].grid(which='both', axis='both')
            ax[1].grid(which='both', axis='both')   
        fig.suptitle('Original microphone ICS Frequency Response')

    # plt.show()

    if calibration:
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
        # Add microphone
        ###############################################################################################################
        # define the locations of the microphones
        mic_loc = [2.45, 2.8, 1.3]
        # finally place the array in the room
        room.add_microphone(loc=mic_loc)

        ###############################################################################################################
        # Run the simulations (this will also build the RIR automatically)
        ###############################################################################################################
        signals = []
        for i in range(num_of_mics):
            room.simulate()
            s_ = room.mic_array.signals
            s = s_[0] / 100
            mic_signal = s

            if add_noise:
                noise = pink_noise(len(s))
                Es = sum(np.power(s[signal_range[0]:signal_range[1]], 2))
                En = sum(np.power(noise[signal_range[0]:signal_range[1]], 2))
                alpha = np.sqrt(Es/(10**(SNR/10)*En))
                mic_signal = s + alpha*noise

            signals.append(mic_signal)

        min_val = -80
        max_val = -30
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.specgram(s/ np.abs(mic_signal).max(), NFFT=256, Fs=fs, vmin=min_val, vmax=max_val)
        plt.title("Original Signal")
        plt.subplot(2, 1, 2)
        plt.specgram(mic_signal/ np.abs(mic_signal).max(), NFFT=256, Fs=fs, vmin=min_val, vmax=max_val)
        plt.title("Noisy Signal")

        print("Mic signal length: ", len(signals[0]))
        print("Noise: ", add_noise)
        if add_noise:
            print("SNR: ", SNR)
            print("Denoise: ", denoise)
        
        if args.method == "ism":
            # measure the reverberation time
            rt60 = room.measure_rt60()
            print("The desired RT60 was {}".format(rt60_tgt))
            print("The measured RT60 is {}".format(rt60[0, 0]))

            # if plot_figure:
                # plot the RIRs
                # select = None  # plot all RIR
                # # select = (2, 0)  # uncomment to only plot the RIR from mic 2 -> src 0
                # # select = [(0, 0), (2, 0)]  # only mic 0 -> src 0, mic 2 -> src 0
                # fig, axes = room.plot_rir(select=select, kind="ir")  # impulse responses
                # fig, axes = room.plot_rir(select=select, kind="tf")  # transfer function
                # fig, axes = room.plot_rir(select=select, kind="spec")  # spectrograms
            
        ###############################################################################################################
        # Create microphone filtered signals
        ###############################################################################################################
        mic_signals = []
        for i in range(len(signals)):
            mic_signal = signal.sosfilt(ICS[i], signals[i])
            calibsig = mic_signal[signal_range[0]:signal_range[1]]
            if denoise:
                calibsig = nr.reduce_noise(y=calibsig, y_noise=mic_signal[0:signal_range[0]], stationary=True, sr=fs, n_std_thresh_stationary=n_std_thresh_stationary)
            mic_signals.append(calibsig)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.specgram(mic_signal/ np.abs(mic_signal).max(), NFFT=256, Fs=fs, vmin=min_val, vmax=max_val)
        plt.title("Noisy Signal")
        plt.subplot(2, 1, 2)
        plt.specgram(calibsig/ np.abs(mic_signal).max(), NFFT=256, Fs=fs, vmin=min_val, vmax=max_val)
        plt.title("Denoise Signal")

        # plot microphone signals
        if plot_figure:
            fig, axs = plt.subplots(len(mic_signals), sharex=True)
            for i, y in enumerate(mic_signals):
                x = np.linspace(0, len(mic_signals[i])/fs, len(mic_signals[i]))
                axs[i].plot(x, y)
                axs[i].set_xlim(0, len(mic_signals[i])/fs)
                axs[i].set_ylim(-0.05, 0.05)
                if i == len(mic_signals) - 1:
                    axs[i].set_xlabel('Time [s]')
        # plt.show()

        for i, y in enumerate(mic_signals):
            scipy.io.wavfile.write(f"simulation_calibrations/calibration_signal{i}_{SNR}_{rt60_tgt}.wav", fs, y)
        print("Microphone signal files written.")

    if evaluate:
        ###############################################################################################################
        # Load calibration filter
        ###############################################################################################################
        G = np.loadtxt(f"simulation_calibrations/G_{SNR}_{rt60_tgt}.txt").view(complex)

        # load calibration filter from Matlab
        # g = scipy.io.loadmat('../g.mat')
        # G = fft(g['g'], axis=1)

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
            fig, ax = plt.subplots(2, sharex=True)
            w = np.linspace(0, fs*(K-1)/K, K)
            for icsfr_c in ICSFR_C:
                Yabs = abs(icsfr_c)
                ax[0].plot(w, 20 * np.log10(np.maximum(Yabs, 1e-5)))
                ax[1].plot(w, np.angle(icsfr_c))
                ax[0].set_ylabel('Amplitude [dB]')
                ax[1].set_xlabel('Frequency [Hz]')
                ax[1].set_ylabel('Phase [rad]')
                ax[0].axis((10, fs/2, -100, 10))
                ax[1].axis((10, fs/2, -5, 5))
                ax[0].grid(which='both', axis='both')
                ax[1].grid(which='both', axis='both')   
            fig.suptitle('Calibrated microphone ICS Frequency Response')

        ###############################################################################################################
        # calculate evaluation criteria
        ###############################################################################################################
        print("calibrated SNR: ", SNR, " RT60: ", rt60_tgt)
        calculate_evaluation_metrics(ICSFR_C)


    if plot_figure:
            plt.tight_layout()
            plt.show()
    