"""Test of the diagonal term in the rate theorem and of the conjecture on Phi_{A,B}
(Sections 8 and 12).  Chain DB: a_n = n^3 (c = 1, alpha = 3, kappa = 0), diagonal
b_n = 2 n^(3/2) (beta = 3/2 < alpha - 1 = 2).  Eigenvalues of H_0 from odd-size truncations
with the diagonal (GPU Sturm counts, sturm_diag.py), three sizes, extrapolation in
n^(-p), p = alpha - 1 - beta (the diagonal tail).  Reported, as functions of N = (lambda/2)^(1/3):
  r_A  = Phi_A(lambda_j) reduced mod pi to (-pi/2, pi/2]   (theorem: O(N^(1+beta-alpha)) = O(N^-1/2) here)
  pred = first-order prediction (B0/(4 c alpha)) B((alpha-beta-1)/(2 alpha), 1/2) N^(1+beta-alpha)
  r_AB = Phi_{A,B}(lambda_j) reduced mod pi                 (conjecture: O(1/N))
"""
import sys, os, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import sturm_diag as sd, sturm
import cupy as cp
from scipy.special import beta as Beta
from scipy.integrate import quad

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
c, al, B0, be = 1.0, 3.0, 2.0, 1.5
Ia = 0.5 * Beta((al - 1) / (2 * al), 0.5)
pred_c = B0 / (4 * c * al) * Beta((al - be - 1) / (2 * al), 0.5)

def PhiA(lam):
    return (lam / (2 * c)) ** (1 / al) * Ia

def PhiAB(lam):
    A = lambda x: c * x ** al
    Bf = lambda x: B0 * x ** be
    # largest root X of 2A = lam - B, then integrand pi/2 below X (lam - B > 2A there for large lam)
    from scipy.optimize import brentq
    X = brentq(lambda x: 2 * A(x) + Bf(x) - lam, 1e-9, (lam / (2 * c)) ** (1 / al) + 1)
    f = lambda x: math.asin(max(-1.0, min(1.0, (lam - Bf(x)) / (2 * A(x)))))
    v1, _ = quad(f, X, 2 * X, limit=400, epsabs=1e-13, epsrel=1e-13)
    v2, _ = quad(f, 2 * X, math.inf, limit=400, epsabs=1e-13, epsrel=1e-13)
    return X * math.pi / 2 + v1 + v2

class Wrap:
    def __init__(self, g): self.g = g
    def count(self, lam, dlast): return self.g.count(lam, 0.0)

lines = []
def say(s):
    print(s, flush=True); lines.append(s)

Ns = [10, 14, 20, 28, 40, 56, 80, 113, 160]
js = sorted(set(int(round(N * Ia / math.pi)) for N in Ns))
p = al - 1 - be      # slowest tail: the diagonal part of the shear weights, sum_k b_k k^(-alpha) ~ n^(1+beta-alpha)
n0 = 2 ** 20
sizes = [n0, 2 * n0, 4 * n0]
k = np.arange(1, 8 * n0 + 9, dtype=np.float64)
a_full = c * k ** al
b_full = B0 * np.arange(0, 8 * n0 + 9, dtype=np.float64) ** be
Ls = []
t0 = time.time()
for m in sizes:
    size = 2 * m + 1
    g = sd.GPUCounterDiag(a_full[:size] ** 2, b_full[:size], size)
    Nj = (np.array(js) * math.pi + math.pi / 2) / Ia
    guess = 2 * c * Nj ** al
    c0 = int(cp.asnumpy(g.count(np.array([1e-6])))[0])
    ks = np.array([int(cp.asnumpy(g.count(np.array([x])))[0]) - c0 for x in guess])
    lam, _ = sturm.positive_eigs(Wrap(g), ks, guess * 0.25, guess * 4.0, 0.0, 1e-6)
    Ls.append(lam)
    del g; cp.get_default_memory_pool().free_all_blocks()
L = np.array(Ls)
q = 2.0 ** (-p)
red = lambda x: (x + math.pi / 2) % math.pi - math.pi / 2
rows = []
say(f"chain DB: a_n = n^3, b_n = 2 n^1.5; sizes {sizes}; prediction constant (B0/(4 c alpha)) B(...) = {pred_c:.6f}")
for i in range(L.shape[1]):
    rA = np.array([red(PhiA(L[s, i])) for s in range(3)])
    rAB = np.array([red(PhiAB(L[s, i])) for s in range(3)])
    eA = (rA[2] - q * rA[1]) / (1 - q); eAB = (rAB[2] - q * rAB[1]) / (1 - q)
    N = (L[2, i] / (2 * c)) ** (1 / al)
    rows.append(dict(N=N, lam=[float(x) for x in L[:, i]], rA_raw=[float(x) for x in rA], rAB_raw=[float(x) for x in rAB], rA=float(eA), rAB=float(eAB)))
    say(f"N={N:8.3f}  N^(alpha-1-beta) r_A = {eA * N ** (al - 1 - be):+.5f} (pred {pred_c:+.5f})   N r_AB = {eAB * N:+.5f}"
        f"   size spread {abs(rA[2] - rA[1]):.1e}")
say(f"time {time.time() - t0:.1f}s")
json.dump(rows, open(os.path.join(OUT, "diag_balanced_results.json"), "w"), indent=1)
open(os.path.join(OUT, "diag_balanced_output.txt"), "w").write(chr(10).join(lines) + chr(10))
