import numpy as np
import scipy.signal
from scipy.signal import stft, istft, find_peaks
import matplotlib.pyplot as plt

def filter_complex_signal_stft(signal, G):
    K = len(G)
    # Compute STFT
    f, t, Ystft = stft(signal, nperseg=K, return_onesided=False)

    # Apply the filter in the frequency domain
    filtered_Zxx_real = Ystft.real * G[:, np.newaxis]
    filtered_Zxx_imag = Ystft.imag * G[:, np.newaxis]

    # Reconstruct the filtered signal
    _, filtered_signal = istft(filtered_Zxx_real+1j*filtered_Zxx_imag, nperseg=K, input_onesided=False)

    return filtered_signal[:len(signal)]

def filter_complex_signal(X, sos):
    # Filter the real and imaginary parts separately
    filtered_real = scipy.signal.sosfilt(sos, X.real)
    filtered_imag = scipy.signal.sosfilt(sos, X.imag)

    # Combine the real and imaginary parts to form the filtered complex signal
    return filtered_real + 1j * filtered_imag

def power_mvdr(theta, X):
    s = np.exp(-2j * np.pi * d * np.arange(X.shape[0]) * np.sin(theta)) # steering vector in the desired direction theta_i
    s = s.reshape(-1,1) # make into a column vector (size 3x1)
    R = (X @ X.conj().T)/X.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
    Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better than a true inverse
    return 1/(s.conj().T @ Rinv @ s).squeeze()

def rms(matrix):
    # Calculate the squared error
    squared_error = (matrix)**2
    # Calculate the mean of squared errors along the third dimension
    mse_along_dimension = np.mean(squared_error, axis=0)
    # Take the square root to get RMSE
    rmse_along_dimension = np.sqrt(mse_along_dimension)
    
    return rmse_along_dimension

if __name__ == "__main__":
    ###############################################################################################################
    # parameters
    ###############################################################################################################
    labels = ["matched", "uncalibrated", "calibrated"]
    N = 10000 # number of samples to simulate
    d = 0.5 # half wavelength spacing
    Nr = 8
    theta_degrees = [-60, -15, 0, 15, 60] # direction of arrival (feel free to change this, it's arbitrary)
    Ns = len(theta_degrees)
    fs = 44100
    trials = 100
    G = np.loadtxt(f"simulation_calibrations/p_{0.0005}/G_{20.0}_{0.215}_{0}.txt").view(complex)
    theta_scan = np.linspace(-0.5*np.pi, 0.5*np.pi, 1000) # 1000 different thetas between -90 and +90 degrees
    plot = False

    ###############################################################################################################
    # microphone ICSs
    ###############################################################################################################
    ICS = []
    for i in np.arange(-np.floor(Nr/2), np.ceil(Nr/2)):
       ics = scipy.signal.iirfilter(17, [1000+2*i*100, 20000-2*i*100], rs=60, btype='band',
                        analog=False, ftype='cheby2', fs=fs,
                        output='sos')
       ics[0,:3] = ics[0,:3] * pow(10,2*i*1/20)
       ICS.append(ics)

    ###############################################################################################################
    # steering vectors
    ###############################################################################################################
    s = np.zeros((Ns,Nr),dtype=complex)
    for i, theta_degree in enumerate(theta_degrees):
        theta = theta_degree / 180 * np.pi # convert to radians
        s_row = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Steering Vector
        s[i,:] = s_row

    errors_CDS = []
    errors_MVDR = []
    errors_MUSIC = []
    for t in range(trials):
        print("trial: ", t)
        ###############################################################################################################
        # create received signals
        ###############################################################################################################
        tx = np.zeros((Ns,N),dtype=complex)
        for i in range(Ns):
            tx_row = np.random.normal(0,1,N)  # Signal of Interest

            # Create a tone to act as the transmitter signal
            # t = np.arange(N)/fs # time vector
            # f_tone = 4083.3
            # tx_row = np.exp(2j * np.pi * f_tone * t)
            tx[i,:] = tx_row

        # Simulate the received signal X through a matrix multiply
        X = s.T @ tx  # dont get too caught up by the transpose, the important thing is we're multiplying the steering vector by the tx signal

        # plt.plot(np.abs(np.asarray(X[1,:]).squeeze())[0:200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
        # plt.plot(np.asarray(X[1,:]).squeeze().real[0:200])
        # plt.plot(np.asarray(X[2,:]).squeeze().real[0:200])
        # plt.show()

        # add noise
        n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
        X = X + 0.01*n

        ###############################################################################################################
        # filter the signals with microphone ICSs
        ###############################################################################################################
        match_signals = np.zeros(X.shape, dtype=complex)
        uncalibrated_signals = np.zeros(X.shape, dtype=complex)
        calibrated_signals = np.zeros(X.shape, dtype=complex)

        for i in range(X.shape[0]):
            match_signals[i,:] = filter_complex_signal(X[i,:], ICS[0])
            uncalibrated_signals[i,:] = filter_complex_signal(X[i,:], ICS[i])
            calibrated_signals[i,:] = filter_complex_signal_stft(uncalibrated_signals[i,:], G[2*i,:])

        signals = [match_signals, uncalibrated_signals, calibrated_signals]

        # signals = [X]
        ###############################################################################################################
        # conventional delay and sum
        ###############################################################################################################
        # print("====================CDS====================")
        results_CDS = []
        error = []
        for i, signal in enumerate(signals):
            result = []
            for theta_i in theta_scan:
                w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Conventional, aka delay-and-sum, beamformer

                # spatial tapering
                # tapering = np.hamming(Nr) # Hamming window function
                # w *= tapering

                signal_weighted = w.conj().T @ signal # apply our weights. remember X is 3x10000
                result.append(np.var(signal_weighted))
            result = (result-np.min(result))/(np.max(result)-np.min(result)) 
            results_CDS.append(result)

            peaks, _ = find_peaks(result.real, height=0.1)
            if peaks.shape[0] == 5:
                error.append(theta_scan[peaks] * 180 / np.pi - theta_degrees)
            # print angle that gave us the max value
            # print(labels[i], np.around(theta_scan[peaks] * 180 / np.pi,3))
            
        errors_CDS.append(error)
        # print()

        if plot:
            fig, ax = plt.subplots(figsize = (7, 7))
            for i, result in enumerate(results_CDS):
                ax.plot(theta_scan*180/np.pi, result.real, label=labels[i]) # lets plot angle in degrees
            ax.set_xlabel("Theta [Degrees]")
            ax.set_ylabel("DOA Metric")
            ax.set_title("CDS")
            ax.grid()
            ax.legend()

        ###############################################################################################################
        # MVDR
        ###############################################################################################################
        # print("===================MVDR===================")
        results_MVDR = []
        error = []
        for i, signal in enumerate(signals):
            result = []
            for theta_i in theta_scan:
                power = power_mvdr(theta_i, signal)
                # power_dB = 10*np.log10(power) # power in signal, in dB so its easier to see small and large lobes at the same time
                result.append(power)
            result = (result-np.min(result))/(np.max(result)-np.min(result)) 
            results_MVDR.append(result)

            peaks, _ = find_peaks(result.real, height=0.1)
            if peaks.shape[0] == 5:
                error.append(theta_scan[peaks] * 180 / np.pi - theta_degrees)
            # print angle that gave us the max value
            # print(labels[i], np.around(theta_scan[peaks] * 180 / np.pi,3))
        # print()
        errors_MVDR.append(error)
        
        if plot:
            fig, ax = plt.subplots(figsize = (7, 7))
            for i, result in enumerate(results_MVDR):
                ax.plot(theta_scan*180/np.pi, result.real, label=labels[i]) # lets plot angle in degrees
            ax.set_xlabel("Theta [Degrees]")
            ax.set_ylabel("DOA Metric")
            ax.set_title("MVDR")
            ax.grid()
            ax.legend()

        ###############################################################################################################
        # MUSIC
        ###############################################################################################################
        # print("==================MUSIC===================")
        num_expected_signals = Ns # Try changing this!

        results_MUSIC = []
        error = []
        for i, signal in enumerate(signals):
            # part that doesn't change with theta_i
            R = np.cov(signal) # Calc covariance matrix. gives a Nr x Nr covariance matrix
            w, v = np.linalg.eig(R) # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
            eig_val_order = np.argsort(np.abs(w)) # find order of magnitude of eigenvalues
            v = v[:, eig_val_order] # sort eigenvectors using this order
            # We make a new eigenvector matrix representing the "noise subspace", it's just the rest of the eigenvalues
            V = np.zeros((Nr, Nr - num_expected_signals), dtype=np.complex64)
            for n in range(Nr - num_expected_signals):
                V[:, n] = v[:, n]

            result = []
            for theta_i in theta_scan:
                sm = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Steering Vector
                sm = sm.reshape(-1,1)
                metric = 1 / (sm.conj().T @ V @ V.conj().T @ sm) # The main MUSIC equation
                metric = np.abs(metric.squeeze()) # take magnitude
                # metric = 10*np.log10(metric) # convert to dB
                result.append(metric)
            result = (result-np.min(result))/(np.max(result)-np.min(result)) 
            results_MUSIC.append(result)

            peaks, _ = find_peaks(result.real, height=0.1)
            if peaks.shape[0] == 5:
                error.append(theta_scan[peaks] * 180 / np.pi - theta_degrees)
            # print angle that gave us the max value
            # print(labels[i], np.around(theta_scan[peaks] * 180 / np.pi,3))
        # print()
        errors_MUSIC.append(error)

        if plot:
            fig, ax = plt.subplots(figsize = (7, 7))
            for i, result in enumerate(results_MUSIC):
                ax.plot(theta_scan*180/np.pi, result.real, label=labels[i]) # lets plot angle in degrees
            ax.set_xlabel("Theta [Degrees]")
            ax.set_ylabel("DOA Metric")
            ax.set_title("MUSIC")
            ax.grid()
            ax.legend()

            plt.show()

    rmse_CDS = rms(np.array(errors_CDS))
    rmse_MVDR = rms(np.array(errors_MVDR))
    rmse_MUSIC = rms(np.array(errors_MUSIC))
    print(rmse_CDS)
    print(rmse_MVDR)
    print(rmse_MUSIC)

    np.savetxt(f"errors_cds.txt", rmse_CDS.view(float))
    np.savetxt(f"errors_mvdr.txt", rmse_MVDR.view(float))
    np.savetxt(f"errors_music.txt", rmse_MUSIC.view(float))
