import numpy as np
from scipy.integrate import odeint


def ode(z, t, Iext, G, Ein, Eex, eps, a, b, A, Bpb, Bbp, vsl):
    vp, vb, up, ub, sp, sb = z
    dzdt = [vp - vp ** 3 / 3 - up + Iext + G * sb * (Ein - vp),  # + Gpb * spb * (Eexin - vb),
            vb - vb ** 3 / 3 - ub + G * sp * (Eex - vb),  # + Gbb * sbp * (Eexin - vp),
            eps * (vp + a - b * up), eps * (vb + a - b * ub),
            A / 2 * (1 + np.tanh(vp / vsl)) * (1 - sp) - Bpb * sp,
            A / 2 * (1 + np.tanh(vb / vsl)) * (1 - sb) - Bbp * sb]
    return dzdt


def calcODE(args, z0, dz0, ddz0, d3z0, d4z0, d5z0, ts=2000, nt=2 ** 10):
    z0 = [z0, dz0, ddz0, d3z0, d4z0, d5z0]
    t = np.linspace(0, ts, nt)
    sol = odeint(ode, z0, t, args)
    return sol, t


def ode1(z, t, Iext, G, Ein, Eex, eps, a, b, A, Bpb, Bbpmin, Bbpmax, vsl, Tmax):
    grate = (Bbpmax - Bbpmin) / Tmax
    return ode(z, t, Iext, G, Ein, Eex, eps, a, b, A, Bpb, Bbpmin + grate * t, vsl)


def calcODE1(args, z0, dz0, ddz0, d3z0, d4z0, d5z0, ts=2000, nt=2 ** 10):
    z0 = [z0, dz0, ddz0, d3z0, d4z0, d5z0]
    t = np.linspace(0, ts, nt)
    Tmax = ts
    args = args + (Tmax,)
    sol = odeint(ode1, z0, t, args)
    return sol, t


def ode2(z, t, Iext, G, Ein, Eex, eps, a, b, A, Bpbmin, Bpbmax, Bbp, vsl, Tmax):
    # почему здесь Ein с Eex поменяны местами?
    grate = (Bpbmax - Bpbmin) / Tmax
    return ode(z, t, Iext, G, Ein, Eex, eps, a, b, A, Bpbmin + grate * t, Bbp, vsl)


def calcODE2(args, z0, dz0, ddz0, d3z0, d4z0, d5z0, ts=2000, nt=2 ** 10):
    z0 = [z0, dz0, ddz0, d3z0, d4z0, d5z0]
    t = np.linspace(0, ts, nt)
    Tmax = ts
    args = args + (Tmax,)
    sol = odeint(ode2, z0, t, args)
    return sol, t
