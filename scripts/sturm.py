"""Sturm-sequence eigenvalue counts for (bordered) truncations of a zero-diagonal
Jacobi matrix, on the GPU (cupy RawKernel, float64) with a numpy CPU reference.

For the size-n truncation (sites 0..n-1) with links a_1..a_{n-1} and a single
diagonal entry `dlast` at the last site, the number of eigenvalues below lam
equals the number of negative pivots of the LDL^T factorisation of (J_n - lam):
    q_0 = d_0 - lam,   q_k = d_k - lam - a_k^2 / q_{k-1}.
Each GPU thread evaluates one lam; one kernel launch is one pass over the
sites, so no launch approaches the Windows display-driver watchdog.
"""
import numpy as np

try:
    import cupy as cp
    _SRC = r'''
    extern "C" __global__
    void sturm(const double* a2, const long long n, const double* lam,
               const double* dlast, const long long nl, long long* cnt) {
        long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= nl) return;
        const double L = lam[i];
        const double dl = dlast[i];
        double q = (n == 1 ? dl : 0.0) - L;
        long long c = (q < 0.0);
        if (q == 0.0) q = -1e-300;
        for (long long k = 1; k < n; ++k) {
            double d = (k == n - 1) ? dl : 0.0;
            q = d - L - a2[k - 1] / q;
            if (q == 0.0) q = -1e-300;
            c += (q < 0.0);
        }
        cnt[i] = c;
    }'''
    _kern = cp.RawKernel(_SRC, "sturm")
    HAVE_GPU = True
except Exception:  # pragma: no cover
    HAVE_GPU = False


def count_cpu(a2, n, lam, dlast):
    """Vectorised over lam; a2[k-1] = a_k^2."""
    lam = np.asarray(lam, dtype=np.float64)
    dlast = np.broadcast_to(np.asarray(dlast, dtype=np.float64), lam.shape)
    q = (dlast if n == 1 else 0.0) - lam
    c = (q < 0).astype(np.int64)
    q = np.where(q == 0, -1e-300, q)
    for k in range(1, n):
        d = dlast if k == n - 1 else 0.0
        q = d - lam - a2[k - 1] / q
        q = np.where(q == 0, -1e-300, q)
        c += (q < 0)
    return c


class GPUCounter:
    def __init__(self, a2_host, n):
        self.n = int(n)
        if len(a2_host) < self.n - 1:
            raise ValueError("not enough links for the requested truncation size")
        self.a2 = cp.asarray(a2_host[: self.n - 1], dtype=cp.float64)

    def count(self, lam, dlast):
        lam = cp.asarray(lam, dtype=cp.float64)
        dl = cp.asarray(np.broadcast_to(np.asarray(dlast, dtype=np.float64), lam.shape), dtype=cp.float64)
        nl = lam.size
        out = cp.empty(nl, dtype=cp.int64)
        threads = 128
        blocks = (nl + threads - 1) // threads
        _kern((blocks,), (threads,), (self.a2, cp.int64(self.n), lam, dl, cp.int64(nl), out))
        return out


def positive_eigs(counter, targets, lo, hi, dlast, lam_small, iters=58):
    """Find the k-th positive eigenvalues (k in targets) by bisection in log(lam).

    counter.count(lam, dlast) -> counts below lam.  lo, hi: initial brackets
    (arrays), widened automatically if they do not bracket.  Returns lam (host)."""
    targets = np.asarray(targets, dtype=np.int64)
    m = targets.size
    c0 = int(cp.asnumpy(counter.count(np.array([lam_small]), dlast))[0])
    lo = np.array(lo, dtype=np.float64)
    hi = np.array(hi, dtype=np.float64)
    for _ in range(60):   # widen brackets until count(lo) - c0 < k <= count(hi) - c0
        clo = cp.asnumpy(counter.count(lo, dlast)) - c0
        chi = cp.asnumpy(counter.count(hi, dlast)) - c0
        bad_lo = clo >= targets
        bad_hi = chi < targets
        if not (bad_lo.any() or bad_hi.any()):
            break
        lo = np.where(bad_lo, lo * 0.5, lo)
        hi = np.where(bad_hi, hi * 2.0, hi)
    llo, lhi = np.log(lo), np.log(hi)
    for _ in range(iters):
        mid = 0.5 * (llo + lhi)
        cm = cp.asnumpy(counter.count(np.exp(mid), dlast)) - c0
        up = cm >= targets
        lhi = np.where(up, mid, lhi)
        llo = np.where(up, llo, mid)
    return np.exp(0.5 * (llo + lhi)), c0


def positive_eigs_ksect(counter, targets, lo, hi, dlast, lam_small, K=256, rounds=9):
    """Same result as positive_eigs, but each round evaluates K log-spaced points per
    target in one kernel launch (K-section instead of bisection), so a round narrows
    the bracket by a factor K.  Nine rounds with K = 256 reach the float64 resolution
    of log(lambda), as 58 bisection steps do, with about seven times fewer passes
    over the sites."""
    targets = np.asarray(targets, dtype=np.int64)
    c0 = int(cp.asnumpy(counter.count(np.array([lam_small]), dlast))[0])
    lo = np.array(lo, dtype=np.float64)
    hi = np.array(hi, dtype=np.float64)
    for _ in range(60):
        clo = cp.asnumpy(counter.count(lo, dlast)) - c0
        chi = cp.asnumpy(counter.count(hi, dlast)) - c0
        bad_lo = clo >= targets
        bad_hi = chi < targets
        if not (bad_lo.any() or bad_hi.any()):
            break
        lo = np.where(bad_lo, lo * 0.5, lo)
        hi = np.where(bad_hi, hi * 2.0, hi)
    llo, lhi = np.log(lo), np.log(hi)
    frac = np.arange(1, K) / K
    for _ in range(rounds):
        pts = llo[:, None] + (lhi - llo)[:, None] * frac[None, :]      # (m, K-1)
        cnt = cp.asnumpy(counter.count(np.exp(pts).ravel(), dlast)).reshape(pts.shape) - c0
        up = cnt >= targets[:, None]                                   # monotone in the point index
        first = np.where(up.any(axis=1), up.argmax(axis=1), K - 1)     # first point with count >= target
        new_hi = np.where(first < K - 1, pts[np.arange(len(pts)), np.minimum(first, K - 2)], lhi)
        new_lo = np.where(first > 0, pts[np.arange(len(pts)), np.maximum(first - 1, 0)], llo)
        llo, lhi = new_lo, new_hi
    return np.exp(0.5 * (llo + lhi)), c0
