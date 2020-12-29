from BP_dynamics import *


def sAnalytics(v):
    N = len(v) // 2
    F = np.fft.fft(v)
    F = np.concatenate([2 * F[0:N], np.zeros(N)])
    return np.fft.ifft(F)


def fftMorlet(t, func, scale, omega0):
    """
    Построение вейвлет-преобразования Морле

    :param t: массив времен
    :param func: функция, для которой проводится вейвлет-преобразование
    :param scale: массив масштабов для вейвлета
    :param omega0: нулевая частота (обычно, 2 * np.pi)
    :return: массив функции с вейвлет-преобразованием
    """
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


def max_wavelet_bootstrap(args, args2, scale, n_iters=100, sol_init=(-1.5, -1.5, 0.5, 0.5, 0.5, 0.5)):
    """
    Параметры масштабирования, на которых достигается максимум вейвлет-функции (для построения графиков гистерезиса)

    :param args: аргументы для CalcODE
    :param args2: аргументы для CalcODE2
    :param scale: массив масштабов для вейвлетов
    :param n_iters: количество шагов для набора статистики
    :return: масштабирующие параметры
    """

    sol_init, t = calcODE(args, *sol_init, ts=4000, nt=2 ** 18)
    sol_init = sol_init[len(sol_init) // 2:]

    ws = None
    for _ in range(n_iters):
        sol, t = calcODE2(args2, *sol_init[np.random.randint(0, len(sol_init))], 2000, 2 ** 15)
        w = fftMorlet(t, sol[:, 0], scale, 2 * np.pi)
        w = np.abs(w)

        if ws is None:
            ws = w[..., np.newaxis]
        else:
            ws = np.concatenate([ws, w[..., np.newaxis]], axis=-1)
    print(ws.shape)
    return scale[np.argmax(np.quantile(ws, 0.75, axis=-1), axis=0)], ws
