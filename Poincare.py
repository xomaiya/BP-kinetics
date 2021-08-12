import numpy as np
import matplotlib.pyplot as plt
from BP_dynamics import calcODE, ode
from scipy.integrate import odeint
from tqdm.auto import trange


def dist(x, x0, n):
    """
    Auxiliary function for point-plane distance calculation

    :param x: point
    :param x0: point through which the plane passes
    :param n: normal to plane
    :return: distance
    """
    return abs(n / np.linalg.norm(n) @ (x - x0))


def poincare(args, parameter, initial_conditions=(-1.5, -1.5, 0.5, 0.5, 0.5, 0.5), ts=4000, nt=2 ** 20, show=True):
    """
    Poincare map plot.
    Для использования этой функции необходимо определить свою динамическую систему,
    задав функцию calcODE(args, *initial_conditions)

    :param args: arguments of the dynamical system
    :return: periods, xs
    """

    xs = []
    periods = []

    if show:
        plt.figure(figsize=(10, 10))

    sol, t = calcODE(args, *initial_conditions, ts=ts, nt=nt)
    sol = sol[-len(sol) // 2:, :]

    x0 = sol[0, :]
    t = t[-len(t) // 2:]
    n = np.array(ode(x0, t[0], *args))
    q, _ = np.linalg.qr(n[:, None], mode='complete')

    period = None

    for i in range(len(sol) - 1):
        x1 = sol[i]
        x2 = sol[i + 1]
        if np.sign(n @ (x2 - x0)) != np.sign(n @ (x1 - x0)):
            c1 = dist(x1, x0, n)
            c2 = dist(x2, x0, n)
            alpha = c2 / (c1 + c2)
            x_new = x1 + alpha * (x2 - x1)
            x = (x_new - x0).dot(q)
            xs.append((parameter, x))
            if show:
                plt.scatter(x[1], x[2])
            if np.linalg.norm(x_new - x0) < 1e-3 and period is None:
                period = t[i] - t[0]
                periods.append((parameter, period, np.linalg.norm(x_new - x0)))
    #             else:
    #                 print(np.linalg.norm(x_new - x0))

    if show:
        plt.show()
    return periods, xs


def bifurcation_diagram(args, Bpbmin, Bpbmax, ylim=(-1, 0.6)):
    """
    Bifurcation diagram plot for BP-system.

    :param args: args of the BP-system
    :param Bpbmin: parameter of the BP-system
    :param Bpbmax: parameter of the BP-system
    :return: xs, periods
    """

    xs = []
    Bpb_list = np.linspace(Bpbmin, Bpbmax, 100)
    Iext, G, Ein, Eex, eps, a, b, A, Bpb, Bbp, vsl = args

    sol, t = calcODE(args, -1.5, -1.5, 0.5, 0.5, 0.5, 0.5, ts=4000, nt=2 ** 25)
    sol = sol[-len(sol) // 2:, :]
    t = t[-len(t) // 2:]

    x0 = sol[0, :]
    n = np.array(ode(x0, t[0], *args))
    q, _ = np.linalg.qr(n[:, None], mode='complete')

    periods = []
    for Bpb in Bpb_list:
        args = (Iext, G, Ein, Eex, eps, a, b, A, Bpb, Bbp, vsl)
        sol, t = calcODE(args, *sol[-1, :], ts=1000, nt=2 ** 15)
        sol = sol[-len(sol) // 2:, :]
        t = t[-len(t) // 2:]

        for i in range(len(sol) - 1):
            x1 = sol[i]
            x2 = sol[i + 1]
            if np.sign(n @ (x2 - x0)) != np.sign(n @ (x1 - x0)):
                c1 = dist(x1, x0, n)
                c2 = dist(x2, x0, n)
                alpha = c2 / (c1 + c2)
                x_new = x1 + alpha * (x2 - x1)
                x = (x_new - x0).dot(q)
                xs.append((Bpb, x[0], x[1], x[2], x[3], x[4], x[5]))
                # if np.linalg.norm(x_new - x0) < 1e-2 and period is None:
                period = t[i] - periods[-1][-1] if len(periods) else 0
                periods.append((Bpb, period, np.linalg.norm(x_new - x0), t[i]))

    plt.figure(figsize=(15, 10))
    plt.scatter([i[0] for i in xs], [i[2] for i in xs], s=10)
    plt.xlabel('$B_{pb}$')

    # plt.ylim(ylim)
    plt.show()

    periods = [i for i in periods if i[1] > 0]

    return periods, xs


def poincare_3D(args, initial_conditions=(-1.5, -1.5, 0.5, 0.5, 0.5, 0.5), ts=4000, nt=2 ** 20):
    """
    Poincare map in 3D.

    :param args: arguments and parameters of dynamical system
    :return: None
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    sol, t = calcODE(args, *initial_conditions, ts=ts, nt=nt)

    sol = sol[-len(sol) // 2:, :]
    ax.plot(xs=sol[:, 0], ys=sol[:, 1], zs=sol[:, 2])
    # T = period(sol[-nt // 2:, :])
    # print(f'T = {T}, {t[T] - t[0]}')
    x0 = sol[0, :]
    t = t[-len(sol) // 2:]
    n = ode(x0, t, *args)

    for i in range(len(sol) - 1):
        x1 = sol[i]
        x2 = sol[i + 1]
        if np.sign(n @ (x1 - x0)) != np.sign(n @ (x2 - x0)):
            c1 = dist(x1, x0, n)
            c2 = dist(x2, x0, n)
            alpha = c2 / (c1 + c2)
            x = x1 + alpha * (x2 - x1)
            ax.scatter(x[0], x[1], x[2])
    plt.show()


def jacobian(z, t, args):
    """
    BP-system jacobian

    :param z: vector (vp, vb, up, ub, sp, sb)
    :param args: arguments of the BP-system
    :return:
    """
    vp, vb, up, ub, sp, sb = z
    Iext, G, Ein, Eex, eps, a, b, A, Bpb, Bbp, vsl = args
    J = np.array([
        [1 - vp ** 2 - G * sb, 0, -1, 0, 0, G * (Ein - vp)],
        [0, 1 - vb ** 2 - G * sp, 0, -1, G * (Eex - vb), 0],
        [eps, 0, -eps * b, 0, 0, 0],
        [0, eps, 0, -eps * b, 0, 0],
        [A / 2 * (1 - sp) / (np.cosh(vp / vsl)) ** 2 / vsl, 0, 0, 0, -A / 2 * (1 + np.tanh(vp / vsl)) - Bpb, 0],
        [0, A / 2 * (1 - sb) / (np.cosh(vb / vsl)) ** 2 / vsl, 0, 0, 0, -A / 2 * (1 + np.tanh(vb / vsl)) - Bbp]
    ])
    return J


def flattened_jacobian(z, t, args):
    return jacobian(z, t, args).flatten()


def calc_ode_jac(args, z0, nt=2 ** 10):
    sol0, t = calcODE(args, *z0, ts=2000, nt=nt)
    ts = period(sol0[-nt // 2:, :])
    t = np.linspace(0, ts, nt)
    ode = flattened_jacobian
    sol = odeint(ode, z0, t, args)
    return sol, t


def jacobian_x(z, x, args):
    J = jacobian(x, args)
    return J @ z


def period(sol):
    x0 = sol[0, :]
    for i in range(100, sol.shape[0]):
        if np.linalg.norm(sol[i, :] - x0) < 5e-3:
            break

    return i + np.argmin(np.linalg.norm(sol[i: int(i * 1.1)] - x0, axis=1))


def DE_monodromy(z, t, Iext, G, Ein, Eex, eps, a, b, A, Bpb, Bbp, vsl):
    vp, vb, up, ub, sp, sb = z[:6]
    M = z[6:].reshape((6, 6))
    args = (Iext, G, Ein, Eex, eps, a, b, A, Bpb, Bbp, vsl)

    dzdt = [vp - vp ** 3 / 3 - up + Iext + G * sb * (Ein - vp),
            vb - vb ** 3 / 3 - ub + G * sp * (Eex - vb),
            eps * (vp + a - b * up),
            eps * (vb + a - b * ub),
            A / 2 * (1 + np.tanh(vp / vsl)) * (1 - sp) - Bpb * sp,
            A / 2 * (1 + np.tanh(vb / vsl)) * (1 - sb) - Bbp * sb]
    dMdt = (jacobian(z[:6], t, args) @ M).reshape(-1)

    return np.concatenate([dzdt, dMdt])


def calcODE_monodromy(args, z0, ts, nt):
    t = np.linspace(0, ts, nt)
    sol = odeint(DE_monodromy, z0, t, args)
    return sol, t


def monodromy(args, initial_conditions, nt=2 ** 25):
    sol, t = calcODE(args, *initial_conditions, 20000, nt)
    T = period(sol[-nt // 2:, :])
    T = t[T] - t[0]
    print(f'T = {T}')

    initial_conditions_M = np.concatenate([sol[-1, :], np.eye(6).reshape(-1)])
    sol_M, t = calcODE_monodromy(args, initial_conditions_M, ts=T, nt=2 ** 20)

    M = sol_M[-1][6:].reshape((6, 6))  # - np.eye(6)
    return M, T


def floquet(args, initial_conditions):
    M, T = monodromy(args, initial_conditions, nt=2 ** 25)
    mus = np.linalg.eigvals(M)[1:]
    print(f'Multipliers: {mus}')
    print()

    print('Проверка условия с дивергенцией: ')
    sol_cycle, t = calcODE(args, *initial_conditions, T, 2 ** 20)
    res = 0
    for i in trange(len(t)):
        res += np.diag(jacobian(sol_cycle[i, :], (i * T) / len(t), args)).sum() * (t[1] - t[0])

    print('These values must be almost the same')
    print(np.product(mus), np.exp(res))

    return mus
