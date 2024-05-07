import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy.signal
from scipy.signal import tf2sos, stft, istft
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
from pyargus.directionEstimation import *

def subspace_decomposition(R):
    # eigenvalue decomposition!
    # This method is specialized for Hermitian symmetric matrices,
    # which is the case since R is a covariance matrix
    w, v = np.linalg.eigh(R)

    # This method (numpy.linalg.eigh) returns the eigenvalues (and
    # eigenvectors) in ascending order, so there is no need to sort Signal
    # comprises the leading eigenvalues Noise takes the rest

    Es = v[..., -1 :]
    ws = w[..., -1 :]
    En = v[..., : -1]
    wn = w[..., : -1]

    return (Es, En, ws, wn)

def compute_correlation_matricesvec(X):
    # change X such that time frames, frequency microphones is the result
    X = np.transpose(X, axes=[2, 1, 0])
    # select frequency bins
    freq_bins = np.arange(5, 60)
    X = X[..., list(freq_bins), :]
    # Compute PSD and average over time frame
    C_hat = np.matmul(X[..., None], np.conjugate(X[..., None, :]))
    # Average over time-frames
    C_hat = np.mean(C_hat, axis=0)
    return C_hat

def compute_spatial_spectrumvec(cross):
    mod_vec = np.transpose(
        np.array(self.mode_vec[self.freq_bins, :, :]), axes=[2, 0, 1]
    )
    # timeframe, frequ, no idea
    denom = np.matmul(
        np.conjugate(mod_vec[..., None, :]), np.matmul(cross, mod_vec[..., None])
    )
    return 1.0 / abs(denom[..., 0, 0])

def music(X):
    """
    Perform MUSIC for given frame in order to estimate steered response
    spectrum.
    """
    num_freq = 55
    M = 4
    # compute steered response
    # self.Pssl = np.zeros((self.num_freq, self.grid.n_points))
    C_hat = compute_correlation_matricesvec(X)
    print(C_hat.shape)
    # subspace decomposition
    Es, En, ws, wn = subspace_decomposition(C_hat[None, ...])
    print(Es.shape)
    # compute spatial spectrum
    identity = np.zeros((num_freq, M, M))
    identity[:, list(np.arange(M)), list(np.arange(M))] = 1
    print(identity.shape)
    cross = identity - np.matmul(Es, np.moveaxis(np.conjugate(Es), -1, -2))
    print(cross.shape)
    self.Pssl = self._compute_spatial_spectrumvec(cross)

def spherical_to_cartesian(phi, theta, r=1):
    """
    Convert spherical coordinates (phi, theta) to Cartesian coordinates.
    phi: azimuth angle in radians
    theta: colatitude angle in radians
    r: radius of the sphere (default is 1)
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    # print(x, y, z)
    return x, y, z

def plot_point_on_sphere(R, phi, theta):
    """
    Plot a point on a 3D sphere given its spherical coordinates (phi, theta).
    phi: azimuth angle in radians
    theta: colatitude angle in radians
    """

    center = [np.mean(R[0]), np.mean(R[1]), np.mean(R[2])]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

    # Plot microphone array
    ax.scatter(R[0]-center[0], R[1]-center[1], R[2]-center[2], color = 'c')

    # Plot point
    x_p, y_p, z_p = spherical_to_cartesian(phi, theta)
    # ax.scatter(x_p, y_p, z_p, color='r', s=100)

    # Plot arrow from center to point
    ax.quiver(0, 0, 0, x_p, y_p, z_p, color='m', arrow_length_ratio=0.1, linewidths=3)

    # Plot axes
    ax.quiver(0, 0, 0, 0.3, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0.3, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 0.3, color='b', arrow_length_ratio=0.1)

    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])

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

def filter_signal_stft(signal, G):
    K = len(G)
    # Compute STFT
    f, t, Ystft = stft(signal, nperseg=K, return_onesided=False)

    # Apply the filter in the frequency domain
    filtered_Zxx = Ystft * G[:, np.newaxis]

    # Reconstruct the filtered signal
    _, filtered_signal = istft(filtered_Zxx, nperseg=K, input_onesided=False)

    return np.real(filtered_signal)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a dry sound sample. Saves it into `./examples/samples`."
    )

    parser.add_argument('-p', '--plot',
                    action='store_true')  # on/off flag

    args = parser.parse_args()

    ###############################################################################################################
    # Parameters
    ###############################################################################################################
    fs = 44100
    K = 1024
    num_of_mics = 4
    c = 343.0  # speed of sound
    freq_bins = np.arange(5, 60)  # FFT bins to use for estimation

    # The desired reverberation time and dimensions of the room
    SNR = 20.0  # signal-to-noise ratio
    trials = 5
    dim = 2
    room_dim = np.r_[10.0, 10.0]
    d = 0.126
    rt60_tgt = 0.2

    azimuth = 50.0 / 180.0 * np.pi  # 60 degrees
    distance = 1.0  # 3 meters

    wav_directory = "../signal/"
    file_names = ["white_noise_15s.wav"]
    nfft = 256
    plot_figure = args.plot

    ###############################################################################################################
    # simulate microphone ICS
    ###############################################################################################################
    ICS = []
    ICS.append(scipy.signal.iirfilter(17, [400, 21650], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=fs,
                       output='sos'))
    ICS.append(scipy.signal.iirfilter(17, [500, 21550], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=fs,
                       output='sos'))
    ICS.append(scipy.signal.iirfilter(17, [600, 21450], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=fs,
                       output='sos'))
    ICS.append(scipy.signal.iirfilter(17, [700, 21350], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=fs,
                       output='sos'))
    ICS[1][0,:3] = ICS[1][0,:3] * pow(10,-1/20)
    ICS[2][0,:3] = ICS[2][0,:3] * pow(10,3/20)
    ICS[3][0,:3] = ICS[3][0,:3] * pow(10,-7/20)

    ICSFR_O = []
    u = scipy.signal.unit_impulse(K)
    for i in range(len(ICS)):
        ics_imp = scipy.signal.sosfilt(ICS[i], u)
        Y = fft(ics_imp)
        ICSFR_O.append(Y)

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
    # Create the room
    ###############################################################################################################
    fs, source_signals = extract_measurements(wav_directory, file_names, 0)
    
    sigma2 = 10 ** (-SNR / 10) / (4.0 * np.pi * distance) ** 2
    sigma2 = 0
    room = pra.AnechoicRoom(dim, fs=fs, sigma2_awgn=sigma2)

    # place the source in the room
    source_location = room_dim / 2 + distance * np.r_[np.cos(azimuth), np.sin(azimuth)]
    # source_signal = np.random.randn((nfft // 2 + 1) * nfft)
    room.add_source(source_location, signal=source_signals[0])

    ###############################################################################################################
    # Add microphones
    ###############################################################################################################
    # define the locations of the microphones
    # We use a rectangular array with radius 4.2 cm # and 16 microphones
    R = pra.linear_2D_array(room_dim / 2, 4, 0, 0.126)
    mic_locs = R
    print(mic_locs)
    # finally place the array in the room
    room.add_microphone_array(mic_locs)

    # Creating figure
    if plot_figure:
        fig, ax = plt.subplots(figsize = (10, 7))
    
        # Creating plot
        ax.scatter(mic_locs[0], mic_locs[1], color = "green")
        ax.scatter(source_location[0], source_location[1], color = "red")

        ax.axis('equal')

        ax.set_title("simple 2D scatter plot")
        ax.plot()

    ###############################################################################################################
    # Run the simulations (this will also build the RIR automatically)
    ###############################################################################################################
    room.simulate()
    S = room.mic_array.signals

    ###############################################################################################################
    # Create microphone filtered signals
    ###############################################################################################################
    match_signals = []
    uncalibrated_signals = []
    calibrated_signals = []

    G = np.loadtxt(f"simulation_calibrations/p_{0}/G_{35.0}_{0.28}_{0}.txt").view(complex)

    for i in range(S.shape[0]):
        s = S[i,:]/100
        match_signals.append(scipy.signal.sosfilt(ICS[0], s))

        mic_signal = scipy.signal.sosfilt(ICS[i], s)
        uncalibrated_signals.append(mic_signal)

        filtered_mic_signal = filter_signal_stft(mic_signal,G[i,:])
        calibrated_signals.append(filtered_mic_signal)

    ################################
    # Compute the STFT frames needed
    X_matched = np.array(
        [
            pra.transform.stft.analysis(signal, nfft, nfft // 2).T
            for signal in match_signals
        ]
    )

    X_uncalibrated = np.array(
        [
            pra.transform.stft.analysis(signal, nfft, nfft // 2).T
            for signal in uncalibrated_signals
        ]
    )

    X_calibrated = np.array(
        [
            pra.transform.stft.analysis(signal, nfft, nfft // 2).T
            for signal in calibrated_signals
        ]
    )

    X = [X_matched, X_uncalibrated, X_calibrated]

    # ##############################################
    # # Now we can test all the algorithms available
    # # algo_names = sorted(pra.doa.algorithms.keys())
    # # algo_names.remove("FRIDA")

    # doa = pra.doa.algorithms['MUSIC'](R, fs, nfft, c=c, dim=2)

    # for i in range(len(X)):
    #     # print(X[i][:,0:10,0])
    #     doa.locate_sources(X[i], freq_bins=freq_bins)
    #     print(X[i].shape)
    #     print(doa.azimuth_recon.shape)
    #     azimuth = doa.azimuth_recon if doa.azimuth_recon < 0.5*np.pi else 2.0*np.pi-doa.azimuth_recon
    #     doa.plot_individual_spectrum()
    #     plt.show()
    #     # colaltitude = doa.colatitude_recon if doa.colatitude_recon < 0.5*np.pi else np.pi-doa.colatitude_recon
    #     print("  Recovered azimuth:", azimuth / np.pi * 180.0, "degrees")
    #     # print("  Recovered colatitude:", colaltitude  / np.pi * 180.0, "degrees")

    music(X_matched)
    
    if plot_figure:
        plt.show()


