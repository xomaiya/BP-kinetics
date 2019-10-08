import numpy as np
from scipy.integrate import odeint

def ode(z, t, Iext, G, Ein, Eex, eps, a, b, A, Bpb, Bbp, vsl):
    vp, vb, up, ub, sbp, spb = z
    dzdt = [vp - vp **3 / 3 - up + Iext + G * spb * (Eex - vp), # + Gpb * spb * (Eexin - vb),
            vb - vb **3 / 3 - ub + G * sbp * (Ein - vb), # + Gbb * sbp * (Eexin - vp),
            eps * (vp + a - b * up), eps * (vb + a - b * ub),
            A / 2 * (1 + np.tanh(vp / vsl)) * (1 - sbp) - Bpb * sbp,
            A / 2 * (1 + np.tanh(vb / vsl)) * (1 - spb) - Bbp * spb]
    return dzdt

def calcODE(args, z0, dz0, ddz0, d3z0, d4z0, d5z0, ts = 2000, nt = 10000):
    z0 = [z0, dz0, ddz0, d3z0, d4z0, d5z0]
    t = np.linspace(0, ts, nt)
    sol = odeint(ode, z0, t, args)
    return sol

def ode1(z, t, Iext, G, Ein, Eex, eps, a, b, A, Bpb, Bbpmin, Bbpmax, vsl, Tmax):
    grate = (Bbpmax - Bbpmin) / Tmax
    vp, vb, up, ub, spb, sbp = z
    dzdt = [vp - vp ** 3 / 3 - up + Iext + G * sbp * (Ein - vp), # + Gpb * spb * (Eexin - vb),
            vb - vb ** 3 / 3 - ub + G * spb * (Eex - vb), # + Gbb * sbp * (Eexin - vp),
            eps * (vp + a - b * up), eps * (vb + a - b * ub),
            A / 2 * (1 + np.tanh(vp / vsl)) * (1 - spb) - Bpb * spb,
            A / 2 * (1 + np.tanh(vb / vsl)) * (1 - sbp) - (Bbpmin + grate * t) * sbp]
    return dzdt

def calcODE1(args, z0, dz0, ddz0, d3z0, d4z0, d5z0, ts = 2000, nt = 10000):
    z0 = [z0, dz0, ddz0, d3z0, d4z0, d5z0]
    t = np.linspace(0, ts, nt)
    Tmax = ts
    args = args + (Tmax, )
    sol = odeint(ode1, z0, t, args)
    return sol

def ode2(z, t, Iext, G, Ein, Eex, eps, a, b, A, Bpbmin, Bpbmax, Bbp, vsl, Tmax):
    # почему здесь Ein с Eex поменяны местами?
    grate = (Bpbmax - Bpbmin) / Tmax
    vp, vb, up, ub, sbp, spb = z
    dzdt = [vp - vp **3 / 3 - up + Iext + G * spb * (Ein - vp), # + Gpb * spb * (Eexin - vb),
            vb - vb **3 / 3 - ub + G * sbp * (Eex - vb), # + Gbb * sbp * (Eexin - vp),
            eps * (vp + a - b * up), eps * (vb + a - b * ub),
            A / 2 * (1 + np.tanh(vp / vsl)) * (1 - sbp) - (Bpbmin + grate * t) * sbp,
            A / 2 * (1 + np.tanh(vb / vsl)) * (1 - spb) - Bbp * spb]
    return dzdt

def calcODE2(args, z0, dz0, ddz0, d3z0, d4z0, d5z0, ts = 2000, nt = 10000):
    z0 = [z0, dz0, ddz0, d3z0, d4z0, d5z0]
    t = np.linspace(0, ts, nt)
    Tmax = ts
    args = args + (Tmax, )
    sol = odeint(ode2, z0, t, args)
    return sol
