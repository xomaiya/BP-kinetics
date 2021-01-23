import numpy as np
import matplotlib.pyplot as plt
from BP_dynamics import calcODE, calcODE1, calcODE2
from wavelets import sAnalytics, fftMorlet


def phase_portrait(args, args1=None, vp0=0, vb0=0, up0=0, ub0=0, sbp0=0, spb0=0, ts=2000, nt=2 ** 13):
    """
    Функция отрисовки фазового портрета системы. Есть возможность нарисовать фп
    для медленно меняющегося параметра Bpb (нужно раскомментировать 21 строчку с calcODE1)
    или для медленно меняющегося параметра Bbp (нужно раскомменировать 22 строчку с calcODE2)

    :param args: arguments of the BP-system
    :param args1: расширенные параметры системы, включающие границы изменения параметра B (pb или bp)
    :param vp0, vb0, up0, ub0, sbp0, spb0: fixed initial conditions
    :param ts: time
    :param nt: number of steps
    :return: None
    """
    sol, t = calcODE(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)
    # sol, t = calcODE1(args1, *sol[-1], ts, nt)
    # sol, t = calcODE2(args1, *sol[-1], ts, nt)
    vp = sol[-nt // 2:, 0]
    vb = sol[-nt // 2:, 1]

    plt.figure(figsize=(10, 10))
    plt.plot(vb, vp, 'b')
    plt.xlabel('$v_b$')
    plt.ylabel('$v_p$')
    plt.grid()
    plt.show()


def signal_draw(args, vp0=-1.5, vb0=-1.5, up0=0.5, ub0=0.5, sbp0=0.5, spb0=0.5, ts=2000, nt=2 ** 15):
    """
    Функция отрисовки временных рядов Vp и Vb
    :param args: arguments of the BP-system
    :param vp0, vb0, up0, ub0, sbp0, spb0: fixed initial conditions
    :param ts: time
    :param nt: number of steps
    :return: None
    """
    sol, t = calcODE(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)

    plt.figure(figsize=(15, 5))
    plt.plot(np.linspace(0, ts // 4, nt // 4), sol[-nt // 4:, 0], 'b')
    plt.plot(np.linspace(0, ts // 4, nt // 4), sol[-nt // 4:, 1], 'r')

    plt.xlabel('t')
    plt.ylabel('$v_p$, $v_b$')
    plt.grid()
    plt.show()


def signal_draw1(args, args1, vp0=0, vb0=0, up0=0, ub0=0, sbp0=0, spb0=0, ts=2000, nt=2 ** 20):
    """
    Функция отрисовки временных рядов Vp и Vb при медленном изменении параметра Bbp
    По умолчанию оси подписываются как моменты времени, чтобы это изменить, нужно раскомменировать строки 72-73

    :param args: arguments of the BP-system
    :param args1: расширенные параметры системы, включающие границы изменения параметра Bbp
    :param vp0, vb0, up0, ub0, sbp0, spb0: fixed initial conditions
    :param ts: time
    :param nt: number of steps
    :return: None
    """

    sol, t = calcODE(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)
    sol, t = calcODE1(args1, *sol[-1], ts, nt)

    plt.figure(figsize=(15, 5))
    # plt.plot(np.linspace(args[9], args[10], nt), sol[:, 0], 'b')
    # plt.plot(np.linspace(args[9], args[10], nt), sol[:, 1], 'r')
    plt.plot(np.linspace(0, ts, nt), sol[:, 0], 'b')
    plt.plot(np.linspace(0, ts, nt), sol[:, 1], 'r')

    plt.xlabel('t')
    # plt.xlabel('$B_{bp}$')
    plt.ylabel('$v_p, v_b$')
    plt.grid()
    plt.show()


def signal_draw2(args, args2=None, vp0=0, vb0=0, up0=0, ub0=0, sbp0=0, spb0=0, ts=2000, nt=2 ** 15):
    """
    Функция отрисовки временных рядов Vp и Vb при медленном изменении параметра Bpb.
    По умолчанию оси подписываются как моменты времени, чтобы это изменить, нужно раскомменировать строки 102-103

    :param args: arguments of the BP-system
    :param args2: расширенные параметры системы, включающие границы изменения параметра Bpb
    :param vp0, vb0, up0, ub0, sbp0, spb0: fixed initial conditions
    :param ts: time
    :param nt: number of steps
    :return: None
    """

    sol, t = calcODE(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)
    sol, t = calcODE2(args2, *sol[-1], ts, nt)

    plt.figure(figsize=(15, 5))
    # plt.plot(np.linspace(args2[8], args2[9], nt), sol[:, 0], 'b')
    # plt.plot(np.linspace(args2[8], args2[9], nt), sol[:, 1], 'r')
    plt.plot(np.linspace(0, ts, nt), sol[:, 0], 'b')
    plt.plot(np.linspace(0, ts, nt), sol[:, 1], 'r')

    plt.xlabel('t')
    # plt.xlabel('$B_{pb}$')
    plt.ylabel('$v_p$, $v_b$')
    plt.grid()
    plt.show()


def wavelet_draw(args, args2, scale, vp0=-1.5, vb0=-1.5, up0=0.5, ub0=0.5, sbp0=0.5, spb0=0.5, ts=4000, nt=2 ** 15,
                 Bmin=0.7, Bmax=0.9):
    sol, t = calcODE(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)
    sol, t = calcODE2(args2, *sol[-1], ts, nt)

    vp = sol[:, 0]
    vp = sAnalytics(vp)
    res = 2 * fftMorlet(t, vp, scale, 2 * np.pi)
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.ylabel('ISI')

    ticks = np.array([0, len(res) // 4, len(res) // 2, 3 * len(res) // 4, len(res) - 1])
    plt.yticks(ticks, scale[ticks])
    plt.imshow(np.abs(res), aspect='auto', origin='lower')
    xticks = np.linspace(0, nt, 10)
    xlabels = np.array([round(i, 3) for i in np.linspace(Bmin, Bmax, 10)])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('$B_{pb}$')
    plt.show()


def autocorrelation(args):
    PREC = 8
    T = 2 ** 20
    sol, t = calcODE(args, 0, 0, 0, 0, 0, 0, ts=T, nt=PREC * T)
    sol = sol[500 * PREC:]
    sol = sol - np.mean(sol, axis=0, keepdims=True)

    corrs = []
    for tau in range(0, len(sol) // 2, 100):
        corrs.append(np.mean(sol[:tau] * sol[tau: 2 * tau]))

    plt.figure(figsize=(30, 10))
    plt.plot(np.abs(corrs))
    plt.xlabel("t")
    plt.ylabel("Autocorrelation")
    plt.title(f"Модуль автокорреляционной функции (Bbp=0.1, Bpb=0.1879)")
    plt.show()
