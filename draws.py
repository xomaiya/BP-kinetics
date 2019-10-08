import numpy as np
import matplotlib.pyplot as plt
from BP_dynamics import calcODE, calcODE1, calcODE2
from weiwlets import *

def phase_portrait(args, vp0, vb0, up0, ub0, sbp0, spb0, ts=10, nt=100):
    sol = calcODE1(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)
    t = np.linspace(0, 2000, 2 ** 13)

    vp = sol[:, 0]
    scale = np.linspace(0, 60, 101)
    vp = sAnalytics(vp)

    res = fftMorlet(t, vp, scale, np.pi)
    plt.figure(figsize=(10, 10))
    plt.plot(np.abs(res), abs(vp/t) , 'g')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

def signal_draw(args, vp0, vb0, up0, ub0, sbp0, spb0, ts=10, nt=100):
    sol = calcODE(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)

    plt.figure(figsize=(20, 10))
    plt.plot(np.linspace(0, ts, nt), sol[:, 0], 'b')

    plt.xlabel('t')
    plt.ylabel('v_p')
    plt.grid()
    plt.show()

def signal_draw1(args, vp0, vb0, up0, ub0, sbp0, spb0, ts=10, nt=100):
    sol = calcODE1(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)

    plt.figure(figsize=(20, 10))
    plt.plot(np.linspace(0, ts, nt), sol[:, 0], 'b')
    # plt.plot(np.linspace(0, ts, nt), sol[:, 1], 'r')

    plt.xlabel('t')
    plt.ylabel('v_p')
    plt.grid()
    plt.show()

def signal_draw2(args, vp0, vb0, up0, ub0, sbp0, spb0, ts=10, nt=100):
    sol = calcODE2(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)

    plt.figure(figsize=(20, 10))
    plt.plot(np.linspace(0, ts, nt), sol[:, 0], 'b')
    # plt.plot(np.linspace(0, ts, nt), sol[:, 1], 'r')

    plt.xlabel('t')
    plt.ylabel('v_p')
    plt.grid()
    plt.show()
