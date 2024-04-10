import numpy as np
from scipy import signal
import time

def fast_mul(A, B):
    return np.einsum('ij,jk->ik', A, B)

def covariance(y):
    M, T = y.shape
    channelmeans = np.mean(y, axis=1)
    y = y - channelmeans[:, np.newaxis]
    return fast_mul(y, y.T)/T

def stack_delay_data(y, delay):
    M = y.shape[0]

    M_s = (2 * delay + 1) * M
    y_s = np.zeros((M_s, y.shape[1]))

    for tau in range(-delay, delay+1):
        y_shift = np.roll(y, tau, axis=1)
        if tau < 0:
            y_shift[:, tau:] = 0
        elif tau > 0:
            y_shift[:, :tau] = 0
        y_s[M * (tau + delay) : M * (tau + delay + 1), :] = y_shift

    return y_s, M_s

def ensure_symmetry(X):
    if not np.allclose(X, X.T):
        X = (X + X.T) / 2
    return X

def sort_eigenvectors(D, V):
    permutation = np.argsort(D)[::-1]
    V = V[:, permutation]
    D = D[permutation]
    D = np.diag(D)
    return D, V

def wiener_filter(nsignals, signal_range, delay=15, rank="full", fs=44100):
    s = time.time()
    # check if it's a list
    islist = isinstance(nsignals, list)
    if islist:
        min_length = min(len(arr) for arr in nsignals)
        nsignals = [sig[:min_length] for sig in nsignals]
        nsignals = np.array(nsignals)
    
    yd = nsignals
    y = nsignals[:,signal_range[0]:signal_range[1]]
    d = nsignals[:,:signal_range[0]]

    # low pass filter on noise
    sos = signal.iirfilter(17, fs/4, rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=fs,
                       output='sos')
    d = signal.sosfilt(sos, d, axis=1)

    # stack delayed signals
    Ya, M_s = stack_delay_data(y, delay)
    Yd, M_s = stack_delay_data(d, delay)

    # estimate covariance matrices
    # Ryy = np.cov(Ya)
    # Rdd = np.cov(Yd)

    Ryy = covariance(Ya)
    Rdd = covariance(Yd)

    Ryy = ensure_symmetry(Ryy)
    Rdd = ensure_symmetry(Rdd)

    if rank == "full":
        Q = M_s
    # Q = 4
    D, V = np.linalg.eig(Rdd)
    D, V = sort_eigenvectors(D, V)
    D[:,Q:] = 0
    # print(np.diag(D))

    # calculate wiener filter
    VT_inv = np.linalg.inv(V.T)
    Rdd = np.real(np.dot(np.dot(VT_inv, D), np.linalg.inv(V)))
    W = np.linalg.solve(Ryy, Rdd)
    
    # apply filter
    M, T = yd.shape
    channelmeans = np.mean(yd, axis=1)
    yd = yd - channelmeans[:, np.newaxis]
    yd_s, _ = stack_delay_data(yd, delay)
    orig_chans = slice(delay * M, (delay + 1) * M)
    d = np.dot(W[:, orig_chans].T, yd_s)
    n = yd - d
    n = n + channelmeans[:, np.newaxis]

    if islist:
        result = [np.array(sig) for sig in list(n)]
    else:
        result = n

    e = time.time()
    print(e-s)

    return result
