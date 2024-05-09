import numpy as np
from scipy.linalg import toeplitz
import time
import matplotlib.pyplot as plt
from draw_ssas import draw_ssas

def stack_and_concatenate(matrix, L):
    rows, cols = matrix.shape
    result = []
    for k in range(rows//L):
        y = matrix[k * L:(k + 1) * L, :]
        stacked_col = y.T.reshape(1, -1).T
        result.append(stacked_col)
    return np.hstack(result)

def concatenate_and_stack(matrix, L):
    rows, cols = matrix.shape
    result = []
    for j in range(cols):
        y = matrix[:, j]
        stacked_row = y.T.reshape(-1, L).T
        result.append(stacked_row)
    return np.vstack(result)

def correlate(a, b):
    N = len(a)
    return np.dot(a, b.T.conj())/(N-1)

# def correlate(x, y):
#     N = len(x)
#     X = 1/N * np.fft.fft(x)
#     Y = 1/N * np.fft.fft(y)
#     Rxy = N * np.real(np.fft.ifft(X * np.conj(Y)))  # Crosscorrelation function
#     Rxy = toeplitz(Rxy)
#     return Rxy

def auto_correlation(x):
    return correlate(x, x)

def wiener_filter(nsignals, signal_range, coherent=False, L=32, lambdaW=0.9975):
    s = time.time()

    # check if it's a list
    islist = isinstance(nsignals, list)
    if islist:
        min_length = min(len(arr) for arr in nsignals)
        nsignals = [sig[:min_length] for sig in nsignals]
        nsignals = np.array(nsignals).T

    signals = nsignals[signal_range[0]:signal_range[1],:]
    noise = nsignals[:signal_range[0],:]
    N = nsignals.shape[1]

    if coherent:
        results = []
        for i in range(N):
            nsignals_ = nsignals[:L*(len(nsignals)//L),:].copy()
            noise_ = noise.copy()
            nsignals_[:, [i, 0]] = nsignals_[:, [0, i]]
            noise_[:, [i, 0]] = noise_[:, [0, i]]

            # Estimate correlation matrices
            RVV_est = np.zeros((N * L, N * L))
            K = len(noise_) // L
            for k in range(K):
                v = noise_[k * L:(k + 1) * L, :]
                V = v.T.reshape(1, -1).T
                RVV_est += correlate(V, V)
            RVV_est /= K
            # print()
        
            D, V = np.linalg.eig(RVV_est)
            HE = np.real(V[:,-L:])
            np.savetxt(f"he.txt", HE)
            # print(type(V[0,0]))
            # print(HE)
            # print(D)
            # print(V)
            # # print(len(D))
            print(type(nsignals_[0,0]))
            draw_ssas(nsignals_[:,0],44100)
            # plt.plot(nsignals_[:,0])
            # plt.show()
            Y = stack_and_concatenate(nsignals_, L)
            # print(Y[:,0])
            ZW = np.matmul(HE.T, Y)
            # print(ZW)
            # print(ZW.shape)
            # time.sleep(3)
            results.append(concatenate_and_stack(ZW,L))
            t = concatenate_and_stack(ZW,L)
            print(t.shape)
            print(type(t[0,0]))
            np.savetxt(f"t.txt", t)
            draw_ssas(t[:,0],44100)
            # plt.plot(concatenate_and_stack(ZW,L))
            # plt.plot(t)
            # plt.plot(nsignals_[:,0])
            
            plt.show()
            result = np.hstack(results)
    else:
        RYY_est = np.zeros((N * L, N * L))
        K = len(signals) // L
        for k in range(K):
            y = signals[k * L:(k + 1) * L, :]
            Y = y.T.reshape(1, -1).T
            RYY_est += correlate(Y, Y)
        RYY_est /= K

        RVV_est = np.zeros((N * L, N * L))
        K = len(noise) // L
        for k in range(K):
            v = noise[k * L:(k + 1) * L, :]
            V = v.T.reshape(1, -1).T
            RVV_est += correlate(V, V)
        RVV_est /= K

        # e = time.time()
        # print(e-s)
        if lambdaW == 1:
            signals = signals[:L*(len(signals)//L),:]
            Y = stack_and_concatenate(signals, L)
            HW = np.eye(N*L)-np.linalg.solve(RYY_est, RVV_est)
            ZW = np.matmul(HW.T, Y)
            result = concatenate_and_stack(ZW,L)
        else:
            Zw = []
            RYYW_old = RYY_est
            RVV = RVV_est

            K = len(signals) // L
            for k in range(K):
                # s = time.time()
                y = signals[k * L:(k + 1) * L, :]
                Y = y.T.reshape(1, -1).T
                
                RYY = auto_correlation(Y)

                RYYW = lambdaW * RYYW_old + (1 - lambdaW) * RYY
                RYYW_old = RYYW
                # e = time.time()
                # print(e-s)
                # s = time.time()
                HW = np.eye(N*L)-np.linalg.solve(RYYW, RVV)
                # HW = np.eye(N*L)-np.matmul(np.linalg.inv(RYYW),RVV)
                # e = time.time()
                # print(e-s)
                # s = time.time()
                ZW = np.matmul(HW.T, Y)
                zw = ZW.T.reshape(-1,L).T
                Zw.append(zw)
                # e = time.time()
                # print(e-s)
                # time.sleep(3)
            result = np.vstack(Zw)

    if islist:
        result = [np.array(sig) for sig in list(result.T)]
        
    e = time.time()
    print(e-s)
    return result
