from BP_dynamics import *


def sAnalytics(v):
    N = len(v) // 2
    F = np.fft.fft(v)
    F = np.concatenate([2 * F[0:N], np.zeros(N)])
    return np.fft.ifft(F)


def fftMorlet(t, func, scale, omega0):
    N = len(t) // 2
    F = np.fft.fft(func)
    norm = 2 * np.pi / (t[-1] - t[0])
    omega = np.concatenate([np.arange(0, N + 1, 1), np.arange(-N + 1, 0)]) * norm

    w = np.zeros((len(scale), len(func)), dtype=np.complex)

    for i, scl in enumerate(scale):
        if scl == 0:
            w[0, :] = func * np.exp(-omega0 ** 2 / 2)
            continue
        omega_s = scl * omega
        window = np.exp(-(omega_s - omega0) ** 2 / 2)
        w[i, :] = np.fft.ifft(window * F)
    return w


def max_wavelet_bootstrap(args, scale, n_iters=100):
    arg_for_init_sol = args[0:10] + args[11:]

    sol_init, t = calcODE(arg_for_init_sol, 0, 0, 0, 0, 0, 0, ts=2000, nt=2 ** 13)
    sol_init = sol_init[len(sol_init) // 2:]

    ws = None

    for _ in range(n_iters):
        sol, t = calcODE1(args, *sol_init[np.random.randint(0, len(sol_init))], 2000, 2 ** 12)

        w = fftMorlet(t, sol[:, 0], scale, np.pi)
        w = np.abs(w)

        if ws is None:
            ws = w[..., np.newaxis]
        else:
            ws = np.concatenate([ws, w[..., np.newaxis]], axis=-1)

    return scale[np.argmax(np.quantile(ws, 0.25, axis=-1), axis=0)]
