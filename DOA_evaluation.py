import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.io import wavfile
from statistics import mean
import matplotlib.colors as mcolors

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

def plot_points_on_sphere(R, Azimuths, Colatitudes):
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
    for i, c in zip(range(len(Azimuths)), mcolors.TABLEAU_COLORS.keys()):
        for j in range(len(Azimuths[i])):
            x_p, y_p, z_p = spherical_to_cartesian(Azimuths[i][j] * np.pi / 180, Colaltitudes[i][j] * np.pi / 180)
            ax.scatter(x_p, y_p, z_p, color=c, s=20)

    # Plot arrow from center to point
    # ax.quiver(0, 0, 0, x_p, y_p, z_p, color='m', arrow_length_ratio=0.1, linewidths=3)

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
    Azimuths = []
    Colaltitudes = []
    Elevations = []

    for i in range(5):
        azimuths = []
        colaltitudes = []
        elevations = []
        for j in range(36):
            wav_directory = f'../measurements/DOA/{i+1}/'
            file_name = f'record_pos{j}.wav'
            fs, signals = read_audio_file(wav_directory, file_name, [2, 1, 16, 15, 4, 3, 14, 13, 6, 5, 12 ,11, 8, 7, 10, 9])

            ################################
            # Compute the STFT frames needed
            X = np.array(
                [
                    pra.transform.stft.analysis(signal, nfft, nfft // 2).T
                    for signal in signals
                ]
            )

            doa = pra.doa.algorithms['MUSIC'](R, fs, nfft, c=c, dim=3, mode='near')
            # doa = pra.doa.algorithms['NormMUSIC'](R, fs, nfft, c=c, dim=3, mode='near')
            # doa.locate_sources(X, freq_bins=freq_bins)
            doa.locate_sources(X, freq_range=[2300, 2700])
            azimuth = doa.azimuth_recon
            colaltitude = doa.colatitude_recon if doa.colatitude_recon < 0.5*np.pi else np.pi-doa.colatitude_recon
            azimuths.append(azimuth[0]/ np.pi * 180.0)
            colaltitudes.append(colaltitude[0]/ np.pi * 180.0)
            elevations.append(90 - colaltitude[0]/ np.pi * 180.0)
        
        # unwrap
        azimuths = azimuths - azimuths[0]
        for i,az in enumerate(azimuths):
            k = round((az-i*(-10))/360)
            azimuths[i] = -(azimuths[i] + k*(-360))
        Azimuths.append(azimuths)
        Colaltitudes.append(colaltitudes)
        Elevations.append(elevations)


    plt.figure(1)
    for n in range(len(Azimuths)):
        x = range(0, 10*len(Azimuths[n]), 10)
        y = Azimuths[n]
        plt.plot(x, y, marker='o', label=round(mean(Elevations[n]),2))
        plt.legend()
        plt.title('Azimuth Estimation')
        plt.xlabel('Ground Truth (degree)')
        plt.ylabel('Estimation (degree)')

    plt.figure(2)
    for n in range(len(Azimuths)):
        x = range(0, 10*len(Azimuths[n]), 10)
        y = Azimuths[n] - x
        plt.plot(x, y, marker='o', label=round(mean(Elevations[n]),2))
        plt.legend()
        plt.ylim(-10, 10)
        plt.title('Azimuth Estimation Error')
        plt.xlabel('Azimuth (degree)')
        plt.ylabel('Error (degree)')

    plt.figure(3)
    for n in range(len(Elevations)):
        x = range(0, 10*len(Elevations[n]), 10)
        y = Elevations[n]
        plt.plot(x, y, marker='o')
        plt.title('Elevation')
        plt.xlabel('Azimuth (degree)')
        plt.ylabel('Elevation (degree)')

    R = pra.square_2D_array([5.0, 5.0], 4, 4, 0, 0.042)
    R = np.vstack((R[0], R[1], np.array([5.0 for i in range(16)])))
    plot_points_on_sphere(R, Azimuths, Colaltitudes)

    plt.show()


