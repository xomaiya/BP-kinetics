import numpy as np
import matplotlib.pyplot as plt
from BP_dynamics import *
from wavelets import *


def phase_portrait(args, vp0=0, vb0=0, up0=0, ub0=0, sbp0=0, spb0=0, ts=2000, nt=2 ** 13):
    sol, t = calcODE2(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)
    z = [sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4], sol[:, 5]]
    Iext, G, Ein, Eex, eps, a, b, A, Bpbmin, Bpbmax, Bbp, vsl = args
    dzdt = ode2(z, t, Iext, G, Ein, Eex, eps, a, b, A, Bpbmin, Bpbmax, Bbp, vsl, Tmax=2000)
    vp = sol[:, 0]
    vb = sol[:, 1]
    dvpdt = dzdt[0]
    dvbdt = dzdt[1]

    plt.figure(figsize=(10, 10))
    plt.plot(vp, dvpdt, 'b')
    plt.plot(vb, dvbdt, 'r')
    plt.xlabel('vp')
    plt.ylabel('dvp/dt')
    plt.title(f'Фазовый портрет для потенциала $\dot v_p(v_p)$\n ($Bbp = {args[10]}$, $Bpb = {args[8]}-{args[9]}$)')
    plt.grid()
    plt.show()


def signal_draw(args, vp0=0, vb0=0, up0=0, ub0=0, sbp0=0, spb0=0, ts=2000, nt=2 ** 15):
    sol, t = calcODE(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)

    plt.figure(figsize=(20, 10))
    plt.plot(np.linspace(0, ts, nt), sol[:, 0], 'b')
    plt.plot(np.linspace(0, ts, nt), sol[:, 1], 'r')

    plt.xlabel('t')
    plt.ylabel('v_p')
    plt.grid()
    plt.show()


def signal_draw1(args, vp0=0, vb0=0, up0=0, ub0=0, sbp0=0, spb0=0, ts=2000, nt=2 ** 15):
    sol, t = calcODE1(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)

    plt.figure(figsize=(20, 10))
    plt.plot(np.linspace(0, ts, nt), sol[:, 0], 'b')
    plt.plot(np.linspace(0, ts, nt), sol[:, 1], 'r')

    plt.xlabel('t')
    plt.ylabel('v_p')
    plt.grid()
    plt.show()


def signal_draw2(args, vp0=0, vb0=0, up0=0, ub0=0, sbp0=0, spb0=0, ts=2000, nt=2 ** 15):
    sol, t = calcODE2(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)

    plt.figure(figsize=(20, 10))
    plt.plot(np.linspace(0, ts, nt), sol[:, 0], 'b')
    plt.plot(np.linspace(0, ts, nt), sol[:, 1], 'r')

    plt.xlabel('t')
    plt.ylabel('$v_p$, $v_b$')
    plt.title(f'Решение системы ФХН для $v_p, v_b$ ($Bbp = {args[10]}$, $Bpb = {args[8]}-{args[9]}$)')
    plt.grid()
    plt.show()


def wavelet_draw(args, scale, vp0=0, vb0=0, up0=0, ub0=0, sbp0=0, spb0=0, ts=2000, nt=2 ** 13):
    sol, t = calcODE2(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)
    vp = sol[:, 0]
    vp = sAnalytics(vp)
    res = fftMorlet(t, vp, scale, np.pi)

    plt.figure(figsize=(20, 10))
    plt.title(f'CWT для потенциала $v_p$ ($Bbp = {args[10]}$, $Bpb = {args[8]}-{args[9]}$)')
    plt.xlabel("nt")
    plt.ylabel('a')

    ticks = np.array([0, len(res) // 4, len(res) // 2, 3 * len(res) // 4, len(res) - 1])
    plt.yticks(ticks, scale[ticks])

    plt.imshow(np.abs(res), aspect='auto', origin='lower')

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