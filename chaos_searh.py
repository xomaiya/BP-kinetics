from BP_dynamics import *
import matplotlib.pyplot as plt


def mean_T_vp(vp, t):
    nt = len(t)
    t = t[nt // 4: nt]
    vp = vp[nt // 4: nt]

    # ищем пики колебаний v_p (которые больше нуля)
    # vp[1:-1] = [vp(1),.., vp(n-1)]
    # vp[2:] = [vp(2),.., vp(n)]
    # vp[:-2] = [vp(0),.., vp(n-2)]
    # vp[1:-1] > vp[2:] = [vp(1) > vp(2), vp(2) > vp(3),.., vp(n-1) > vp(n)]

    mask = np.logical_and(vp[1:-1] > 0, np.logical_and(vp[1:-1] > vp[:-2], vp[1:-1] > vp[2:]))
    t_of_peaks = t[1:-1][mask]
    return np.mean(t_of_peaks[1:] - t_of_peaks[:-1])


def T_of_attractors(args, initials_num):
    T = []

    for i in range(initials_num):
        vp0 = 4 * np.random.random() - 2
        vb0 = 4 * np.random.random() - 2
        up0 = 4 * np.random.random() - 2
        ub0 = 4 * np.random.random() - 2
        sbp0 = 4 * np.random.random() - 2
        spb0 = 4 * np.random.random() - 2
        sol, t = calcODE2(args, vp0, vb0, up0, ub0, sbp0, spb0)
        T.append(mean_T_vp(sol[:, 0], t))

    plt.figure(figsize=(10, 10))
    plt.hist(T)
    plt.show()
