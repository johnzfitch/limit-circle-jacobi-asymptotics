"""Check the float64 GPU Sturm bisection against 40-digit mpmath bisection on
small truncations (two chains, 12 eigenvalues each, plain and bordered)."""
import sys, os
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(__file__))
from chains import CHAINS
import sturm

mp.mp.dps = 40
out = os.path.join(os.path.dirname(__file__), "..", "outputs", "validate_sturm_output.txt")
lines = []


def count_mp(a2, n, lam, dlast):
    q = (dlast if n == 1 else 0) - lam
    c = int(q < 0)
    for k in range(1, n):
        d = dlast if k == n - 1 else 0
        q = d - lam - a2[k - 1] / q
        if q == 0:
            q = mp.mpf("-1e-300")
        c += int(q < 0)
    return c


def eig_mp(a2, n, k, lo, hi, dlast, c0):
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    for _ in range(140):
        mid = (lo + hi) / 2
        if count_mp(a2, n, mid, dlast) - c0 >= k:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


worst = 0.0
for name, n, dl in [("B3", 801, 0.0), ("D1", 800, 0.0), ("B3", 801, 3.7e5)]:
    ch = CHAINS[name]
    a = ch["links"](n)
    a2f = a * a
    a2m = [mp.mpf(float(x)) ** 2 for x in a]          # same float64 links, squared exactly
    g = sturm.GPUCounter(a2f, n)
    ks = np.arange(1, 13)
    lam_guess = 2 * ch["c"] * (ks * np.pi / 1.0 + 1.0) ** ch["alpha"]
    lam_g, c0 = sturm.positive_eigs(g, ks, lam_guess * 0.1, lam_guess * 10, dl, 1e-6)
    c0m = count_mp(a2m, n, mp.mpf("1e-6"), mp.mpf(dl))
    for k, lg in zip(ks, lam_g):
        lm = eig_mp(a2m, n, int(k), lg * (1 - 1e-9), lg * (1 + 1e-9), mp.mpf(dl), c0m)
        rel = float(abs(lg - lm) / lm)
        worst = max(worst, rel)
        lines.append(f"{name} n={n} dlast={dl:g} k={k:2d} gpu={lg:.17e} mp={mp.nstr(lm, 20)} rel={rel:.2e}")
lines.append(f"worst relative difference: {worst:.2e}  (c0 gpu/mp agree: checked per chain)")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines[-5:]))
