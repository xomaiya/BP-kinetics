import numpy as np
from BP_dynamics import calcODE


def lyapunov_first(args, vp0=-1.5, vb0=-1.5, up0=0.5, ub0=0.5, sbp0=0.5, spb0=0.5, ts=8000, nt=2 ** 15):
    """
    Function for the first Lyapunov exponent calculation (Benetin's algorithm)

    :param args: arguments of the BP-system
    :param vp0, vb0, up0, ub0, sbp0, spb0: initial conditions
    :param ts: time
    :param nt: number of steps
    :return: first Lyapunov exponent
    """
    sol, t = calcODE(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)
    epsilon = 0.0001
    M = 10000000
    T = .001
    x1 = sol[-1]
    x2 = x1 + np.array([epsilon, 0, 0, 0, 0, 0])
    perturbations = []
    for i in range(M):
        sol1, t = calcODE(args, *x1, ts=T, nt=10)
        sol2, t = calcODE(args, *x2, ts=T, nt=10)
        perturbations.append(np.log(np.linalg.norm(sol2[-1] - sol1[-1]) / epsilon))
        x1 = sol1[-1]
        x2 = x1 + (sol2[-1] - sol1[-1]) / np.linalg.norm(sol2[-1] - sol1[-1]) * epsilon
    return np.mean(perturbations) / T


def random_pert():
    """
    Function for the random perturbation of the vector in R^6 with normalization

    :return: perturbation as a vector in R^6
    """
    v = np.random.randn(6)
    return v / np.linalg.norm(v)


def lyapunov_spectra(args, vp0=-1.5, vb0=-1.5, up0=0.5, ub0=0.5, sbp0=0.5, spb0=0.5, ts=8000, nt=2 ** 15):
    """
    Function for the Lyapunov exponents calculation (Benetin's algorithm)

    :param args: arguments of the BP-system
    :param vp0, vb0, up0, ub0, sbp0, spb0: initial conditions
    :param ts: time
    :param nt: number of steps
    :return: vector of the Lyapunov exponents
    """
    sol, t = calcODE(args, vp0, vb0, up0, ub0, sbp0, spb0, ts, nt)
    epsilon = 0.001
    M = 1000000
    T = .001
    x1 = sol[-1]
    x2 = x1 + random_pert() * epsilon
    x3 = x1 + random_pert() * epsilon
    x4 = x1 + random_pert() * epsilon
    x5 = x1 + random_pert() * epsilon
    x6 = x1 + random_pert() * epsilon
    x7 = x1 + random_pert() * epsilon

    perturbations1 = []
    perturbations2 = []
    perturbations3 = []
    perturbations4 = []
    perturbations5 = []
    perturbations6 = []

    for i in range(M):
        sol1, t = calcODE(args, *x1, ts=T, nt=10)
        sol2, t = calcODE(args, *x2, ts=T, nt=10)
        sol3, t = calcODE(args, *x3, ts=T, nt=10)
        sol4, t = calcODE(args, *x4, ts=T, nt=10)
        sol5, t = calcODE(args, *x5, ts=T, nt=10)
        sol6, t = calcODE(args, *x6, ts=T, nt=10)
        sol7, t = calcODE(args, *x7, ts=T, nt=10)

        perturbations1.append(np.log(np.linalg.norm(sol2[-1] - sol1[-1]) / epsilon))
        perturbations2.append(np.log(np.linalg.norm(sol3[-1] - sol1[-1]) / epsilon))
        perturbations3.append(np.log(np.linalg.norm(sol4[-1] - sol1[-1]) / epsilon))
        perturbations4.append(np.log(np.linalg.norm(sol5[-1] - sol1[-1]) / epsilon))
        perturbations5.append(np.log(np.linalg.norm(sol6[-1] - sol1[-1]) / epsilon))
        perturbations6.append(np.log(np.linalg.norm(sol7[-1] - sol1[-1]) / epsilon))

        x1 = sol1[-1]
        # sol_matrix = np.stack([sol2[-1]], axis=1) - x1[:, None]
        sol_matrix = np.stack([sol2[-1], sol3[-1], sol4[-1], sol5[-1], sol6[-1], sol7[-1]], axis=1) - x1[:, None]
        pert_ortho, _ = np.linalg.qr(sol_matrix)

        x2 = x1 + pert_ortho[:, 0] * epsilon
        x3 = x1 + pert_ortho[:, 1] * epsilon
        x4 = x1 + pert_ortho[:, 2] * epsilon
        x5 = x1 + pert_ortho[:, 3] * epsilon
        x6 = x1 + pert_ortho[:, 4] * epsilon
        x7 = x1 + pert_ortho[:, 5] * epsilon

    return np.array([np.mean(perturbations1),
                     np.mean(perturbations2),
                     np.mean(perturbations3),
                     np.mean(perturbations4),
                     np.mean(perturbations5),
                     np.mean(perturbations6)
                     ]) / T


def lyapunov_first_statistics(args, initials_num):
    """
    Function for a more accurate calculation of the first Lyapunov exponent
    by calculating statistics for various initial conditions.

    :param args: arguments of the BP-system
    :param initials_num: number of different initial conditions
    :return: tuple: mean and standard deviation of the first Lyapunov exponent
    """
    lyaps = []
    for i in range(initials_num):
        vp0 = 4 * np.random.random() - 2
        vb0 = 4 * np.random.random() - 2
        up0 = 4 * np.random.random() - 2
        ub0 = 4 * np.random.random() - 2
        sbp0 = np.random.random()
        spb0 = np.random.random()
        sol, t = calcODE(args, vp0, vb0, up0, ub0, sbp0, spb0, ts=8000, nt=2 ** 15)
        epsilon = 0.0001
        M = 100000
        T = .001
        x1 = sol[-1]
        x2 = x1 + epsilon
        perturbations = []
        for i in range(M):
            sol1, t = calcODE(args, *x1, ts=T, nt=10)
            sol2, t = calcODE(args, *x2, ts=T, nt=10)
            perturbations.append(np.log(np.linalg.norm(sol2[-1] - sol1[-1]) / epsilon))
            x1 = sol1[-1]
            x2 = x1 + (sol2[-1] - sol1[-1]) / np.linalg.norm(sol2[-1] - sol1[-1]) * epsilon
        lyaps.append(np.mean(perturbations) / T)
    return np.mean(lyaps), np.std(lyaps)