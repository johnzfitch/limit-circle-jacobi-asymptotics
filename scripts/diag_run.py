"""Test of the small-diagonal theorem (Theorem 'Small diagonal', Section 6) on the
dimerised chain D1 with the diagonal b_k = 0.5 (1+k)^(1/2) (beta = 1/2 < alpha - 1 - 2|kappa'| = 1.6).

Eigenvalues of odd-size (H_0) and even-size (H_{pi/2}) plain truncations with the
diagonal, at three sizes, extrapolated in n^(-p).  The theorem predicts
Phi(lambda_k) = theta_t (mod pi) + o(1) with theta_0 = kappa' pi/(2(alpha-1)) and
theta_{pi/2} = pi/2 - theta_0, where Phi is the zero-diagonal phase.  First a CPU
cross-check of the GPU counts on a small truncation.
"""
import sys, os, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from chains import CHAINS, Phi, I_alpha
import sturm_diag as sd
import sturm
import cupy as cp

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
ch = CHAINS["D1"]
al, c, kap, kp = ch["alpha"], ch["c"], ch["kappa"], ch["kappap"]
th0 = kp * math.pi / (2 * (al - 1))
B0, BETA = 0.5, 0.5
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

# CPU cross-check of the diagonal kernel
a = ch["links"](4000); b = B0 * np.sqrt(1.0 + np.arange(4001))
n = 2001
lam = np.linspace(-50.0, 3e6, 257)
g = sd.GPUCounterDiag(a * a, b, n)
gc = cp.asnumpy(g.count(lam)); cc = sd.count_cpu(a * a, b, n, lam)
say(f"CPU/GPU count agreement on n={n}, 257 lambdas: {bool(np.all(gc == cc))}")

class Wrap:
    def __init__(self, g): self.g = g
    def count(self, lam, dlast): return self.g.count(lam, 0.0)

res = {}
for t, off, p in ((0.0, th0, al - 1 + 2 * kp), (math.pi / 2, math.pi / 2 - th0, al - 1 - 2 * kp)):
    Ns = [10, 14, 20, 28, 40, 56, 80, 113, 160, 226] if t == 0.0 else [10, 14, 20, 28, 40, 56, 80]
    n0 = 2 ** 17 if t == 0.0 else 2 ** 20
    sizes = [n0, 2 * n0, 4 * n0]
    a_full = ch["links"](8 * n0 + 8)
    b_full = B0 * np.power(1.0 + np.arange(8 * n0 + 9, dtype=np.float64), BETA)
    js = sorted(set(int(round((N * I_alpha(al) + kap * math.pi / (2 * al) - off) / math.pi)) for N in Ns))
    js = [j for j in js if j >= 1]
    Ls = []
    t0 = time.time()
    for m in sizes:
        size = 2 * m + 1 if t == 0.0 else 2 * m + 2
        g = sd.GPUCounterDiag(a_full[:size] ** 2, b_full[:size], size)
        Nj = (np.array(js) * math.pi + off - kap * math.pi / (2 * al) + math.pi / 2) / I_alpha(al)
        guess = 2 * c * np.maximum(Nj, 1.0) ** al
        # target index: the eigenvalue nearest the zero-diagonal level, found by counting
        c0 = int(cp.asnumpy(g.count(np.array([1e-6])))[0])
        ks = np.array([int(cp.asnumpy(g.count(np.array([x])))[0]) - c0 for x in guess])
        lam, _ = sturm.positive_eigs(Wrap(g), ks, guess * 0.25, guess * 4.0, 0.0, 1e-6)
        Ls.append(lam)
        del g; cp.get_default_memory_pool().free_all_blocks()
    L = np.array(Ls)
    q = 2.0 ** (-p)
    rows = []
    for i in range(L.shape[1]):
        r = np.array([Phi(ch, L[s, i])[0] for s in range(3)])
        r = (r - off + math.pi / 2) % math.pi - math.pi / 2       # residual mod pi in (-pi/2, pi/2]
        e23 = (r[2] - q * r[1]) / (1 - q)
        N = Phi(ch, L[2, i])[1]
        rows.append(dict(N=float(N), lam=float(L[2, i]), resid=[float(x) for x in r], resid_extrap=float(e23)))
        say(f"t={t:.4f} N={N:9.3f} lambda={L[2, i]:.6e}  Phi - theta_t (mod pi) = {e23:+.3e}"
            f"   N^1.5 * that = {e23 * N ** 1.5:+.4f}")
    say(f"t={t:.4f} sizes={sizes} p={p:.2f} time={time.time() - t0:.1f}s")
    res[f"t={t:.6f}"] = rows
json.dump(res, open(os.path.join(OUT, "diag_results.json"), "w"), indent=1)
open(os.path.join(OUT, "diag_output.txt"), "w").write(chr(10).join(lines) + chr(10))
