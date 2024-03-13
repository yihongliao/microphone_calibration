import numpy as np
import time

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
    return np.dot(a, b.T.conj())

def auto_correlation(x):
    return correlate(x, x)

def wiener_filter(nsignals, signal_range, L=32, lambdaW=0.9975):
    s = time.time()

    signals = nsignals[signal_range[0]:signal_range[1],:]
    noise = nsignals[:signal_range[0],:]
    N = nsignals.shape[1]

    # Estimate correlation matrices
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
        nsignals = nsignals[:L*(len(nsignals)//L),:]
        Y = stack_and_concatenate(nsignals, L)
        HW = np.eye(N*L)-np.linalg.solve(RYY_est, RVV_est)
        ZW = np.matmul(HW.T, Y)
        result = concatenate_and_stack(ZW,L)
    else:
        Zw = []
        RYYW_old = RYY_est
        RVV = RVV_est

        K = len(nsignals) // L
        for k in range(K):
            # s = time.time()
            y = nsignals[k * L:(k + 1) * L, :]
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

    e = time.time()
    print(e-s)
    return result
