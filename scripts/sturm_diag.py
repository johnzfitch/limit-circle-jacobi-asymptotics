"""Sturm counts for truncations with a full diagonal (opt-in variant of sturm.py,
used only for the small-diagonal test of the dimerised theorem).  Same LDL^T
pivot recursion, q_k = b_k + [k = n-1] dlast - lam - a_k^2 / q_{k-1}, one lam per
GPU thread, float64."""
import numpy as np
import cupy as cp

_SRC = r'''
extern "C" __global__
void sturm_diag(const double* a2, const double* b, const long long n, const double* lam,
                const double dlast, const long long nl, long long* cnt) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nl) return;
    const double L = lam[i];
    double q = b[0] + (n == 1 ? dlast : 0.0) - L;
    long long c = (q < 0.0);
    if (q == 0.0) q = -1e-300;
    for (long long k = 1; k < n; ++k) {
        double d = b[k] + ((k == n - 1) ? dlast : 0.0);
        q = d - L - a2[k - 1] / q;
        if (q == 0.0) q = -1e-300;
        c += (q < 0.0);
    }
    cnt[i] = c;
}'''
_kern = cp.RawKernel(_SRC, "sturm_diag")


def count_cpu(a2, b, n, lam, dlast=0.0):
    lam = np.asarray(lam, dtype=np.float64)
    q = b[0] + (dlast if n == 1 else 0.0) - lam
    c = (q < 0).astype(np.int64)
    q = np.where(q == 0, -1e-300, q)
    for k in range(1, n):
        d = b[k] + (dlast if k == n - 1 else 0.0)
        q = d - lam - a2[k - 1] / q
        q = np.where(q == 0, -1e-300, q)
        c += (q < 0)
    return c


class GPUCounterDiag:
    def __init__(self, a2_host, b_host, n):
        self.n = int(n)
        if len(a2_host) < self.n - 1 or len(b_host) < self.n:
            raise ValueError("not enough entries for the requested truncation size")
        self.a2 = cp.asarray(a2_host[: self.n - 1], dtype=cp.float64)
        self.b = cp.asarray(b_host[: self.n], dtype=cp.float64)

    def count(self, lam, dlast=0.0):
        lam = cp.asarray(lam, dtype=cp.float64)
        nl = lam.size
        out = cp.empty(nl, dtype=cp.int64)
        threads = 128
        blocks = (nl + threads - 1) // threads
        _kern((blocks,), (threads,), (self.a2, self.b, cp.int64(self.n), lam, cp.float64(dlast),
                                      cp.int64(nl), out))
        return out
