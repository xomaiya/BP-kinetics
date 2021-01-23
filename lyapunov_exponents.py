import numpy as np
from BP_dynamics import calcODE


def lyapunov_first(args, initial_cinditions=[-1.5, -1.5, 0.5, 0.5, 0.5, 0.5], ts=8000, nt=2 ** 15):
    """
    Function for the first Lyapunov exponent calculation (Benetin's algorithm)

    :param args: arguments of differential equation
    :param initial_cinditions: list of initial conditions
    :param ts: calculation time
    :param nt: number of calculations steps
    :return: first Lyapunov exponent
    """
    sol, t = calcODE(args, *initial_cinditions, ts=ts, nt=nt)
    epsilon = 0.0001
    M = 1000000
    T = .01
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


def lyapunov_spectra2(args, initial_conditions, ts=8000, nt=2 ** 15):
    """
    Function for the Lyapunov exponents calculation (Benetin's algorithm)

    :param args: arguments of the dynamical system
    :param initial_conditions: list of initial conditions
    :param ts: calculation time
    :param nt: number of calculation steps
    :return: vector of the Lyapunov exponents
    """

    n_dims = len(initial_conditions)

    nt_small = 10
    sol, t = calcODE(args, *initial_conditions, ts=ts, nt=nt)
    epsilon = 0.001
    M = 100000
    T = .01
    x = np.zeros((n_dims + 1, n_dims))

    x[0] = sol[-1]
    for i in range(1, n_dims + 1):
        x[i] = x[0] + random_pert(n_dims) * epsilon

    perturbations = np.zeros((M, n_dims))
    sols = np.zeros((n_dims + 1, nt_small, n_dims))

    for i in range(M):
        for j in range(0, n_dims + 1):
            sols[j], _ = calcODE(args, *x[j], ts=T, nt=nt_small)

        for j in range(1, n_dims + 1):
            perturbations[i, j - 1] = np.log(np.linalg.norm(sols[j][-1] - sols[0][-1]) / epsilon)

        x[0] = sols[0, -1]
        sol_matrix = np.stack([sols[j, -1] for j in range(1, n_dims + 1)], axis=1) - x[0][:, None]
        pert_ortho, _ = np.linalg.qr(sol_matrix)

        for j in range(1, n_dims + 1):
            x[j] = x[0] + pert_ortho[:, j - 1] * epsilon

    return np.mean(perturbations, 0) / T


def lyapunov_first_statistics(args, initials_num):
    """
    Function for a more accurate calculation of the first Lyapunov exponent
    by calculating statistics for various initial conditions.
    For customization you have to define the interval of random initial conditions (like "it's for BP-system").

    :param args: arguments of the BP-system
    :param initials_num: number of different initial conditions
    :return: tuple: mean and standard deviation of the first Lyapunov exponent
    """
    lyaps = []
    for i in range(initials_num):
        vp0, vb0, up0, ub0 = 4 * np.random.random(4) - 2  # it's for BP-system
        sbp0, spb0 = np.random.random(2)  # it's for BP-system
        init_conds = [vp0, vb0, up0, ub0, sbp0, spb0]
        lyap = lyapunov_first(args=args, initial_cinditions=init_conds, ts=8000, nt=2 ** 15)
        lyaps.append(lyap)
    return np.mean(lyaps), np.std(lyaps)
