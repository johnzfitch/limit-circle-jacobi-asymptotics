"""Independent recomputation of the headline numbers by finite truncations.

For each chain and extension parameter t we compute positive eigenvalues of
  * the odd-size truncation (t = 0, the extension H_0),
  * the even-size truncation (t = pi/2, the extension H_{pi/2}),
  * the bordered odd-size truncation with last diagonal entry
        d = varpi tan(t) / P_{2m}^2          (generic t),
which imposes the boundary condition (cos t, sin t) of the equal-amplitude
basis at the cut.  No phase, lift, frame, Bessel or Airy function enters.
Each eigenvalue is computed at three truncation sizes n, 2n, 4n and
extrapolated in n^(-p) with the exponent p of the neglected tail.

Output: outputs/truncation_results.json and outputs/truncation_output.txt.
"""
import sys, os, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from chains import CHAINS, Phi, I_alpha
import sturm
import cupy as cp
from scipy.special import gamma as G
from scipy.optimize import brentq

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "outputs")
NCAP = 2 ** 22                    # largest truncation size used (B3, SQ6m1, D1 at t = 0 used 2**26, D1 at t = pi/2 used 2**24; slow-tail cases took over an hour each on the shared GPU)
N_GRID = [10, 14, 20, 28, 40, 56, 80, 113, 160, 226, 320]


def tail_exponent(ch, t):
    al, kp = ch["alpha"], ch.get("kappap", 0.0)
    if abs(kp) < 1e-15:
        return al - 1.0
    if t == 0.0:
        return al - 1 + 2 * kp
    if abs(t - math.pi / 2) < 1e-15:
        return al - 1 - 2 * kp
    return al - 1 - 2 * abs(kp)


def log_varpi_and_P2(a, m_cut, kappap):
    """L_m = log(A_m P_{2m}^2 N_m^{2 kappa'}) for m <= m_cut, extrapolated limit log(varpi)
    (Richardson in 1/N_m with powers 1, 2, 3) and log P_{2m_cut}^2."""
    # a[k-1] = a_k.  P_{2m}^2 = prod_{i=1}^m (a_{2i-1}/a_{2i})^2
    la = np.log(a)
    steps = 2.0 * (la[0:2 * m_cut:2] - la[1:2 * m_cut:2])      # i = 1..m_cut
    logP2 = np.concatenate([[0.0], np.cumsum(steps)])           # logP2[m] = log P_{2m}^2
    def L(m):
        Nm = 2 * m + 1
        return la[2 * m] + logP2[m] + 2 * kappap * math.log(Nm)  # a_{2m+1} = a[2m]
    ms = [m_cut // 8, m_cut // 4, m_cut // 2, m_cut]
    xs = np.array([1.0 / (2 * m + 1) for m in ms])
    ys = np.array([L(m) for m in ms])
    V = np.vstack([np.ones(4), xs, xs ** 2, xs ** 3]).T
    coef = np.linalg.solve(V, ys)
    return coef[0], logP2[m_cut], ys


def run_case(name, ch, t, log):
    al, c = ch["alpha"], ch["c"]
    p = tail_exponent(ch, t)
    # choose the largest N levels for which the tail at n = NCAP/4 is below ~1e-4 rad
    n_small = NCAP // 4
    Ns = [N for N in N_GRID if 2 * N ** al * n_small ** (-p) / p < 1e-4]
    # smallest base size with tail <= 1e-4 at the largest N, rounded to a power of two
    Nmax = max(Ns)
    n_need = (2 * Nmax ** al / p / 1e-4) ** (1.0 / p)
    n0 = int(2 ** math.ceil(math.log2(max(n_need, 2 ** 16))))
    n0 = min(n0, NCAP // 4)
    sizes = [n0, 2 * n0, 4 * n0]
    a_full = ch["links"](8 * n0 + 8)
    kp = ch.get("kappap", 0.0)
    generic = not (t == 0.0 or abs(t - math.pi / 2) < 1e-15)
    # predicted offset theta_t (mod pi) for the level guess
    if ch["kind"] == "dimer":
        th0 = kp * math.pi / (2 * (al - 1))
        off = th0 if t == 0.0 else math.pi / 2 - th0
    else:
        off = t
    jshift = 0 if t == 0.0 else 1          # k-th positive eigenvalue has j = k - jshift
    js = sorted(set(int(round((N * I_alpha(al) + ch["kappa"] * math.pi / (2 * al) - off) / math.pi)) for N in Ns))
    js = [j for j in js if j >= 1]
    ks = np.array([j + jshift for j in js])
    lam_by_size = []
    t0 = time.time()
    for n_half in sizes:
        if t == 0.0 or generic:
            m = n_half
            size = 2 * m + 1                 # sites 0..2m
        else:
            m = n_half
            size = 2 * m + 2                 # sites 0..2m+1
        a = a_full[: size]
        a2 = a * a
        dl = 0.0
        if generic:
            logvarpi, logP2, _ = log_varpi_and_P2(a_full, m, kp)
            dl = math.tan(t) * math.exp(logvarpi - logP2)
        g = sturm.GPUCounter(a2, size)
        Nj = (np.array(js) * math.pi + off - ch["kappa"] * math.pi / (2 * al) + math.pi) / I_alpha(al)
        lam_guess = 2 * c * np.maximum(Nj, 1.0) ** al
        lam, c0 = sturm.positive_eigs_ksect(g, ks, lam_guess * 0.25, lam_guess * 4.0, dl, 1e-6)   # B3, SQ6m1 and D1 at t = 0 were run with the bisection positive_eigs (identical results, sturm.py)
        lam_by_size.append(lam)
        del g
        cp.get_default_memory_pool().free_all_blocks()
    el = time.time() - t0
    L = np.array(lam_by_size)                       # shape (3, levels)
    # extrapolate Phi residuals in n^-p using pairs (n0,2n0) and (2n0,4n0)
    rows = []
    for i, j in enumerate(js):
        r = []
        for s in range(3):
            ph, N = Phi(ch, L[s, i])
            r.append(ph - j * math.pi)
        r = np.array(r)
        q = 2.0 ** (-p)
        e12 = (r[1] - q * r[0]) / (1 - q)
        e23 = (r[2] - q * r[1]) / (1 - q)
        ph, N = Phi(ch, L[2, i])
        rows.append(dict(j=j, N=float(N), lam=[float(x) for x in L[:, i]], r=[float(x) for x in r],
                         r_extrap=float(e23), r_extrap_err=float(abs(e23 - e12))))
    log.append(f"[{name} t={t:.6f}] sizes(m)={sizes} tail p={p:.3f} levels={len(js)} time={el:.1f}s")
    return dict(chain=name, t=t, sizes=sizes, p=p, rows=rows)


CASE_DIR = os.path.join(OUT, "trunc_cases")


def case_list():
    order = ["B3", "SQ6m1", "D1", "BD6", "RS", "B2", "SQ5m0", "SQ5m3", "D2", "B1"]
    cases = []
    for name in order:
        ts = [0.0, math.pi / 2]
        if name in ("B2", "B3", "D1"):
            ts.append(1.0)
        for t in ts:
            cases.append((name, t))
    return cases


def case_file(name, t):
    return os.path.join(CASE_DIR, f"{name}_t{t:.6f}.json")


def run_one(name, t):
    log = []
    res = run_case(name, CHAINS[name], t, log)
    os.makedirs(CASE_DIR, exist_ok=True)
    json.dump(res, open(case_file(name, t), "w"), indent=1)
    print(log[-1], flush=True)


def main():
    """Driver: each case runs in its own subprocess (the GPU is shared with other
    jobs on this machine; a crashed case is retried up to twice).  Finished cases
    are kept in outputs/trunc_cases/ and skipped on a rerun."""
    import subprocess
    for name, t in case_list():
        if os.path.exists(case_file(name, t)):
            continue
        for attempt in range(3):
            r = subprocess.run([sys.executable, "-u", __file__, "--case", name, repr(t)])
            if r.returncode == 0 and os.path.exists(case_file(name, t)):
                break
            print(f"case {name} t={t} attempt {attempt} failed with code {r.returncode}", flush=True)
    res = [json.load(open(case_file(n, t))) for n, t in case_list() if os.path.exists(case_file(n, t))]
    json.dump(res, open(os.path.join(OUT, "truncation_results.json"), "w"), indent=1)
    lines = [f"[{r['chain']} t={r['t']:.6f}] sizes(m)={r['sizes']} tail p={r['p']:.3f} levels={len(r['rows'])}" for r in res]
    open(os.path.join(OUT, "truncation_output.txt"), "w").write(chr(10).join(lines) + chr(10))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--case":
        run_one(sys.argv[2], float(sys.argv[3]))
    else:
        main()
