import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.integrate import odeint
from BP_dynamics import calcODE, ode


def dist(x, x0, n):
    """
    Вспомогательная функция, вычисляющая расстояние от точки до плоскости (Пуанкаре)

    :param x: произвольная точка
    :param x0: точка, через которую проходит плоскость
    :param n: нормаль к плоскости
    :return: расстояние
    """
    return abs(n / np.linalg.norm(n) @ (x - x0))


def poincare(args, parameter, show=True):
    """
    Отрисовка отображения Пуанкаре.

    :param args: arguments of the BP-system
    :return: periods, xs
    """

    xs = []
    periods = []

    if show:
        plt.figure(figsize=(10, 10))

    sol, t = calcODE(args, -1.5, -1.5, 0.5, 0.5, 0.5, 0.5, ts=4000, nt=2 ** 20)
    # sol, t = calcODE(args, 0, 0, 0, 0, 0, 0, ts=4000, nt=2 ** 15)
    sol = sol[-len(sol) // 2:, :]

    x0 = sol[0, :]
    t = t[-len(t) // 2:]
    n = np.array(ode(x0, t[0], *args))
    q, _ = np.linalg.qr(n[:, None], mode='complete')

    period = None

    for i in range(len(sol) - 1):
        x1 = sol[i]
        x2 = sol[i + 1]
        # if np.sign(n @ (x1 - x0)) != np.sign(n @ (x2 - x0)):
        if np.copysign((n @ (x1 - x0)), (n @ (x2-x0))) != (n @ (x1-x0)):
            c1 = dist(x1, x0, n)
            c2 = dist(x2, x0, n)
            alpha = c2 / (c1 + c2)
            x_new = x1 + alpha * (x2 - x1)
            x = (x_new - x0).dot(q)
            xs.append((parameter, x[0], x[1], x[2], x[3], x[4], x[5]))
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
    Bifurcation diagram plot

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
            if np.copysign((n @ (x1 - x0)), (n @ (x2 - x0))) != (n @ (x1 - x0)):
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


def poincare_3D(args):
    """
    Poincare map in 3D

    :param args: arguments and parameters of the BP-system
    :return: None
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    sol, t = calcODE(args, -1.5, -1.5, 0.5, 0.5, 0.5, 0.5, ts=4000, nt=2 ** 20)
    sol = sol[-len(sol) // 2:, :]
    ax.plot(xs=sol[:, 0], ys=sol[:, 1], zs=sol[:, 2])

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


def jacobian(z, args):
    """
    BP-system jacobian

    :param z: vector (vp, vb, up, ub, sp, sb)
    :param args: arguments of the BP-system
    :return:
    """
    vp, vb, up, ub, sp, sb = z
    Iext, G, Ein, Eex, eps, a, b, A, Bpb, Bbp, vsl = args
    J = np.array([
        [1 - vp ** 2 - G * sp, 0, -1, 0, 0, G * (Ein - vp)],
        [0, 1 - vb ** 2 - G * sb, 0, -1, G * (Eex - vb), 0],
        [eps, 0, -eps * b, 0, 0, 0],
        [0, eps, 0, -eps * b, 0, 0],
        [A / 2 * (1 - sp) * vsl / np.cosh(vp / vsl) ** 2, 0, 0, 0, -A / 2 * (1 + np.tanh(vp / vsl) - Bpb), 0],
        [0, A / 2 * (1 - sb) * vsl / np.cosh(vb / vsl) ** 2, 0, 0, 0, -A / 2 * (1 + np.tanh(vb / vsl) - Bbp)]
    ])
    return J


def jacobian_x(z, x, args):
    J = jacobian(x, args)
    return J @ z


def period(sol):
    x0 = sol[0, :]
    for i in range(100, sol.shape[0]):
        if np.linalg.norm(sol[i, :] - x0) < 5e-3:
            return i
    return None


def monodromy(args):
    ts = 2000
    nt = 2 ** 20
    sol, t = calcODE(args, -1.5, -1.5, 0.5, 0.5, 0.5, 0.5, ts, nt)
    T = period(sol[-nt//2:, :])
    print(f'T = {T}')
    sol = sol[-T:, :]
    delta_t = t[-T:] - t[-T-1:-1]

    Z = np.identity(6)
    M = np.zeros((6, 6))
    for j in range(6):
        z = Z[:, j]
        for i in range(0, T):
            z += jacobian_x(z, sol[i, :], args) * delta_t[i]
        M[:, j] = z
    return M

def floquet(args):
    M = monodromy(args)
    return np.linalg.eigvals(M)


# def multiplicators(args, ):
#     sol, t = calcODE(args, -1.5, -1.5, 0.5, 0.5, 0.5, 0.5, ts=2000, nt=2 ** 20)
#     z = sol[-1, :]
#
#     J = jacobian(z, args)
#     monodromy = sc.linalg.expm(J)
#
#     return np.linalg.eigvals(monodromy)





