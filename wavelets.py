import numpy as np

def sAnalytics(v):
    N = len(v)  // 2
    F = np.fft.fft(v)
    F = np.concatenate([2 * F[0:N], np.zeros(N)])
    return np.fft.ifft(F)

def fftMorlet(t, func, scale, omega0):
    N = len(t) // 2
    F = np.fft.fft(func)
    norm = 2 * np.pi / (t[-1] - t[0])
    omega = np.concatenate([np.arange(0, N + 1, 1), np.arange(-N + 1, 0)]) * norm
    w = np.zeros((len(scale), len(func)))

    for i, scl in enumerate(scale):
        if scl == 0:
            w[0, :] = func * np.exp(-omega0 ** 2 / 2)
            continue
        omega_s = scl * omega
        window = np.exp(-(omega_s - omega0) ** 2 / 2)
        w[i, :] = np.fft.ifft(window * F)
    return w