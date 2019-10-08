from draws import *

Iext = 0.5
G = 0.5
Eex = 0
Ein = -5
eps = 0.3
a = 0.5
b = 0.8
A = 1
Bpb = 0.19
Bbp = 0.08
vsl = 0.1

args = (Iext, G, Ein, Eex, eps, a, b, A, Bpb, Bbp, vsl)
signal_draw(args, vp0=0, vb0=0, up0=0, ub0=0, sbp0=0, spb0=0, ts=100, nt=1000000)
