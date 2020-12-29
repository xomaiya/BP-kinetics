import numpy as np
import matplotlib.pyplot as plt
from BP_dynamics import calcODE, calcODE2


def mean_T_vp(vp, t):
    """
    Function for mean of period calculation

    :param vp: time series as an array
    :param t: time
    :return: mean of period for vp time series
    """

    nt = len(t)
    t = t[nt // 5: nt]
    vp = vp[nt // 5: nt]

    mask = np.logical_and(vp[1:-1] > 0.1, np.logical_and(vp[1:-1] > vp[:-2], vp[1:-1] > vp[2:]))
    t_of_peaks = t[1:-1][mask]
    return np.mean(t_of_peaks[1:] - t_of_peaks[:-1])


def T_of_attractors(args, initials_num):
    """
    Calculate and plot of the histogram of the limit cycles periods
    :param args: arguments of the BP-system
    :param initials_num: number of different initial conditions
    :return: list of limit cycle periods
    """
    T = []
    vp_vb_T = []

    for i in range(initials_num):
        ts = 2000
        nt = 2 ** 15
        vp0 = 4 * np.random.random() - 2
        vb0 = 4 * np.random.random() - 2
        up0 = 4 * np.random.random() - 2
        ub0 = 4 * np.random.random() - 2
        sbp0 = np.random.random()
        spb0 = np.random.random()

        sol, t = calcODE2(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)
        T_one = mean_T_vp(sol[:, 0], t) / 2
        print(f'T = {T_one}')
        print(f'Parameters: {vp0, vb0}')
        T.append(T_one)
        vp_vb_T.append((vp0, vb0, T_one))
        plt.figure(figsize=(15, 5))
        plt.plot(np.linspace(0, ts, nt//4), sol[-nt//4:, 0], 'b')

        plt.xlabel('t')
        plt.ylabel('$v_p$')
        plt.grid()
        plt.show()

    plt.figure(figsize=(10, 10))
    plt.hist(T)
    plt.show()

    return T


def map_of_multistability(vars, args, vp0=-1.5, vb0=-1.5, up0=0.5, ub0=0.5, sbp0=0.5, spb0=0.5, ts=2000, nt=2 ** 15):
    """
    Plotting of the map of limit cycle periods for various initial conditions

    :param vars: string of variables for map calculations -- "vv", "uu" or "ss" (it will be varied)
    :param args: arguments of the BP-system
    :param vp0, vb0, up0, ub0, sbp0, spb0: fixed initial conditions
    :param ts: time
    :param nt: number of steps
    :return: array of limit cycle periods
    """
    T = []

    if vars == 'ss':
        initials = np.arange(0, 1, 0.05)
    else:
        initials = np.arange(-2, 2, 0.1)

    for var0_0 in initials:
        T_var1 = []
        for var1_0 in initials:

            if vars == 'vv':
                vp0 = var0_0
                vb0 = var1_0
            elif vars == 'uu':
                up0 = var0_0
                ub0 = var1_0
            else:
                sbp0 = var0_0
                spb0 = var1_0

            sol, t = calcODE(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)
            T_one = mean_T_vp(sol[:, 0], t) / 2
            print(f'T = {T_one}')
            print(f'var_p0 = {var0_0}')
            print(f'var_b0 = {var1_0}')
            T_var1.append(T_one)
        T.append(T_var1)
    return T
