import numpy as np
import time

def correlate(a, b):
    return np.dot(a, b.T.conj())

def auto_correlation(x):
    return correlate(x, x)

def wiener_filter(nsignals, noise, L=64, lambdaW=0.9975):
    s = time.time()
    N = nsignals.shape[1]
    U = np.zeros((L, N * L))
    U[:L, :L] = np.eye(L)

    # Estimate correlation matrices
    RYY_est = np.zeros((N * L, N * L))
    K = len(nsignals) // L
    for k in range(K):
        y = nsignals[k * L:(k + 1) * L, :]
        Y = y.reshape(-1, 1)
        RYY_est += correlate(Y, Y)
    RYY_est /= K

    RVV_est = np.zeros((N * L, N * L))
    K = len(noise) // L
    for k in range(K):
        v = noise[k * L:(k + 1) * L, :]
        V = v.reshape(-1, 1)
        RVV_est += correlate(V, V)
    RVV_est /= K

    Zw = []
    RYYW_old = RYY_est
    RVV = RVV_est

    # e = time.time()
    # print(e-s)

    K = len(nsignals) // L
    for k in range(K):
        # s = time.time()
        y = nsignals[k * L:(k + 1) * L, :]
        Y = y.reshape(-1, 1)
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
        zw = ZW.reshape(L,-1)
        Zw.append(zw)
        # e = time.time()
        # print(e-s)
        # time.sleep(3)
    e = time.time()
    print(e-s)
    return np.vstack(Zw)
