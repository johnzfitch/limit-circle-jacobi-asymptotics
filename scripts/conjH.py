"""Numerical test of the conjecture on leading-order dimerisation (Section 12).

Links a_n = c_e n^alpha (1 + k_e/n) for even n, c_o n^alpha (1 + k_o/n) for odd n,
b = 0.  The matrix is limit point with discrete spectrum, so eigenvalues of large
truncations converge (beyond the outer turning point |o - e| = lambda the tail has
no spectrum below lambda).  We compute the positive eigenvalues of the truncation
of size n and of size 2n, and evaluate Phi_r(lambda_j)/pi - j with |o|, |e| (the
conjecture predicts -1/2 + O(1/j)) and with o, e themselves.
CPU only: scipy's tridiagonal eigensolver on matrices of size <= 12000 (seconds).
"""
import numpy as np, math
from scipy.linalg import eigvalsh_tridiagonal
from scipy.integrate import quad
from scipy.optimize import brentq

def links(n, al, ce, co, ke, ko):
    k = np.arange(1, n + 1, dtype=float)
    even = (np.arange(1, n + 1) % 2 == 0)
    return np.where(even, ce * k**al * (1 + ke / k), co * k**al * (1 + ko / k))

def Phi_r(lam, al, ce, co, ke, ko, absval=True):
    o = lambda y: co * y**al * (1 + ko / y)
    e = lambda y: ce * y**al * (1 + ke / y)
    def f(x):
        oo, ee = o(2*x + 1), e(2*x + 2)
        if absval:
            oo, ee = abs(oo), abs(ee)
        if oo == 0 or ee == 0:
            return math.pi if lam > abs(oo) + abs(ee) else 0.0
        q = (oo*oo + ee*ee - lam*lam) / (2*oo*ee)
        return math.acos(max(-1.0, min(1.0, q)))
    # integrand vanishes once |o - e| > lam; find that point
    xmax = 1.0
    while abs(abs(o(2*xmax+1)) - abs(e(2*xmax+2))) <= lam * 1.0001 or xmax < 2:
        xmax *= 1.5
    pts = [x for x in ((-ko - 1) / 2, (-ke - 2) / 2) if -0.5 < x < xmax]   # sign changes of o, e
    for g in (lambda x: abs(o(2*x+1)) + abs(e(2*x+2)) - lam, lambda x: abs(abs(o(2*x+1)) - abs(e(2*x+2))) - lam):
        xs = np.linspace(-0.5 + 1e-9, xmax, 4000)
        v = [g(x) for x in xs]
        for i in range(len(xs) - 1):
            if v[i] * v[i+1] < 0:
                pts.append(brentq(g, xs[i], xs[i+1]))
    val, err = quad(f, -0.5, xmax, points=sorted(pts) if pts else None, limit=400, epsabs=1e-12, epsrel=1e-12)
    return val

chains = {"H1": (2.5, 1.0, 1.7, 0.2, -0.4), "H2": (2.0, 1.3, 1.0, -0.5, 0.3), "H3": (3.0, 1.0, 0.6, 0.0, 0.0)}
for name, par in chains.items():
    al = par[0]
    for n in (6000, 12000):
        a = links(n - 1, *par)
        ev = eigvalsh_tridiagonal(np.zeros(n), a)
        pos = np.sort(ev[ev > 1e-4])   # truncation end states sit at |lambda| < 1e-7 (exponentially small in exact arithmetic)
        if n == 6000:
            ev6 = pos
    js = [1, 2, 5, 10, 20, 50, 100, 150, 200]
    print(f"{name}: alpha={al}, c_e={par[1]}, c_o={par[2]}, kappa_e={par[3]}, kappa_o={par[4]};"
          f" zero eigenvalue present: {bool(np.any(np.abs(ev) < 1e-12))}")
    for j in js:
        lam = pos[j - 1]
        rel = abs(ev6[j - 1] / lam - 1)
        r_abs = Phi_r(lam, *par) / math.pi - j
        r_raw = Phi_r(lam, *par, absval=False) / math.pi - j
        print(f"  j={j:4d} lambda={lam:.6e} (size change {rel:.1e})  Phi_r/pi - j = {r_abs:+.5f}   with signed o, e: {r_raw:+.5f}")
