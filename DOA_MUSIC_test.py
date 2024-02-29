# -*- coding: utf-8 -*-
"""
DOA Algorithms
==============

This example demonstrates how to use the DOA object to perform direction of arrival
finding in 2D using one of several algorithms

- MUSIC [1]_
- SRP-PHAT [2]_
- CSSM [3]_
- WAVES [4]_
- TOPS [5]_
- FRIDA [6]_

.. [1] R. Schmidt, *Multiple emitter location and signal parameter estimation*, 
    IEEE Trans. Antennas Propag., Vol. 34, Num. 3, pp 276--280, 1986

.. [2] J. H. DiBiase, J H, *A high-accuracy, low-latency technique for talker localization 
    in reverberant environments using microphone arrays*, PHD Thesis, Brown University, 2000

.. [3] H. Wang, M. Kaveh, *Coherent signal-subspace processing for the detection and 
    estimation of angles of arrival of multiple wide-band sources*, IEEE Trans. Acoust., 
    Speech, Signal Process., Vol. 33, Num. 4, pp 823--831, 1985

.. [4] E. D. di Claudio, R. Parisi, *WAVES: Weighted average of signal subspaces for 
    robust wideband direction finding*, IEEE Trans. Signal Process., Vol. 49, Num. 10, 
    2179--2191, 2001

.. [5] Y. Yeo-Sun, L. M. Kaplan, J. H. McClellan, *TOPS: New DOA estimator for wideband 
    signals*, IEEE Trans. Signal Process., Vol. 54, Num 6., pp 1977--1989, 2006

.. [6] H. Pan, R. Scheibler, E. Bezzam, I. Dokmanić, and M. Vetterli, *FRIDA:
    FRI-based DOA estimation for arbitrary array layouts*, Proc. ICASSP,
    pp 3186-3190, 2017

In this example, we generate some random signal for a source in the far field
and then simulate propagation using a fractional delay filter bank
corresponding to the relative microphone delays.

Then we perform DOA estimation and compare the errors for different algorithms

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.io import wavfile

import pyroomacoustics as pra
from pyroomacoustics.doa import circ_dist

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

    plt.show()

def read_audio_file(wav_directory, file_name, channel_maps):
    signals = []
    wav_file_path = wav_directory + file_name
    print('read wav file: ', wav_file_path)
    sample_rate, signal = wavfile.read(wav_file_path)

    for ch in channel_maps:
        signals.append(signal[:, ch-1] / np.iinfo(np.int16).max)

    return sample_rate, signals

if __name__ == "__main__":
    ######
    simulation = True

    #######################
    # algorithms parameters
    c = 343.0  # speed of sound
    fs = 44100  # sampling frequency
    nfft = 256  # FFT size
    freq_bins = np.arange(5, 60)  # FFT bins to use for estimation

    # We use a rectangular array with radius 4.2 cm # and 16 microphones
    R = pra.square_2D_array([5.0, 5.0], 4, 4, 0, 0.042)
    R = np.vstack((R[0], R[1], np.array([5.0 for i in range(16)])))
    # print(R)

    signals = []

    if simulation:
        # Location of original source
        azimuth = 30.0 / 180.0 * np.pi  # 60 degrees
        colaltitude = 85.0 / 180.0 * np.pi  # 60 degrees
        distance = 3.0  # 3 meters
        dim = 3  # dimensions (2 or 3)
        room_dim = np.r_[10.0, 10.0, 10.0]

        # Use AnechoicRoom or ShoeBox implementation. The results are equivalent because max_order=0 for both.
        # The plots change a little because in one case there are no walls.
        use_anechoic_class = True

        print("============ Using anechoic: {} ==================".format(use_anechoic_class))

        
        SNR = 20.0  # signal-to-noise ratio
        # compute the noise variance
        sigma2 = 10 ** (-SNR / 10) / (4.0 * np.pi * distance) ** 2
        # sigma2 = 0


        # Create an anechoic room
        if use_anechoic_class:
            aroom = pra.AnechoicRoom(dim, fs=fs, sigma2_awgn=sigma2)
        else:
            aroom = pra.ShoeBox(room_dim, fs=fs, max_order=0, sigma2_awgn=sigma2)

        # add the source
        # source_location = room_dim / 2 + distance * np.r_[0, np.cos(azimuth), np.sin(azimuth)]
        source_location = room_dim / 2 + distance * np.r_[np.sin(colaltitude)*np.cos(azimuth), np.sin(colaltitude)*np.sin(azimuth), np.cos(colaltitude)]
        # source_location = [8, 6, 9]
        source_signal = np.random.randn((nfft // 2 + 1) * nfft)
        aroom.add_source(source_location, signal=source_signal)
        aroom.add_microphone_array(R)

        # Creating figure
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        # Creating plot
        ax.scatter3D(R[0], R[1], R[2], color = "green")
        ax.scatter3D(source_location[0], source_location[1], source_location[2], color = "red")
        # Plot axes
        ax.quiver(0, 0, 0, 1.5, 0, 0, color='r', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1.5, 0, color='g', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1.5, color='b', arrow_length_ratio=0.1)

        ax.axes.set_xlim3d(left=0, right=10) 
        ax.axes.set_ylim3d(bottom=0, top=10) 
        ax.axes.set_zlim3d(bottom=0, top=10) 
        plt.title("simple 3D scatter plot")
        

        # run the simulation
        aroom.simulate()

        signals = aroom.mic_array.signals
    else:
        fs, _signals = read_audio_file('', 'record_pos1.wav', [2, 1, 16, 15, 4, 3, 14, 13, 6, 5, 12 ,11, 8, 7, 10, 9])
        signals = _signals


    ################################
    # Compute the STFT frames needed
    X = np.array(
        [
            pra.transform.stft.analysis(signal, nfft, nfft // 2).T
            for signal in signals
        ]
    )
    # print(X.shape)

    ##############################################
    # Now we can test all the algorithms available
    # algo_names = sorted(pra.doa.algorithms.keys())
    # algo_names.remove("FRIDA")
    algo_names = ['MUSIC']
    # candidate_azimuth = np.array([30, 45, 60, 180])
    # candidate_colatitude  = np.pi/2*np.ones(1)

    for algo_name in algo_names:
        # Construct the new DOA object
        # the max_four parameter is necessary for FRIDA only
        doa = pra.doa.algorithms[algo_name](R, fs, nfft, c=c, dim=3, max_four=4)

        # this call here perform localization on the frames in X
        doa.locate_sources(X, freq_bins=freq_bins)

        azimuth = doa.azimuth_recon
        colaltitude = doa.colatitude_recon if doa.colatitude_recon < 0.5*np.pi else np.pi-doa.colatitude_recon

        # doa.polar_plt_dirac()
        # plt.title(algo_name)

        # doa.azimuth_recon contains the reconstructed location of the source
        print(algo_name)
        print("  Recovered azimuth:", azimuth / np.pi * 180.0, "degrees")
        print("  Recovered colatitude:", colaltitude  / np.pi * 180.0, "degrees")
        # print("  Error:", circ_dist(azimuth, doa.azimuth_recon) / np.pi * 180.0, "degrees")
        plot_point_on_sphere(R, azimuth, colaltitude)

    plt.show()