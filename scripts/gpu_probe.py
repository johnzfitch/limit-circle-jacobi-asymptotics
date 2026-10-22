"""Bounded GPU/CPU comparison for the Sturm-count pass (AGENTS.md requirement).

Workload: chain B3, truncation size n = 2*10^6, 64 values of lam, one full
pass over the sites.  Timings include packing, host-to-device transfer of the
links, synchronisation and device-to-host transfer of the counts.
"""
import sys, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from chains import CHAINS
import sturm

out = os.path.join(os.path.dirname(__file__), "..", "outputs", "gpu_probe_output.txt")
ch = CHAINS["B3"]
n = 2_000_000
a = ch["links"](n)
a2 = a * a
lam = 2 * ch["c"] * np.linspace(20, 200, 64) ** ch["alpha"]

t0 = time.perf_counter()
c_cpu = sturm.count_cpu(a2, n, lam, 0.0)
t_cpu = time.perf_counter() - t0

import cupy as cp
g = sturm.GPUCounter(a2, n); cp.cuda.Device().synchronize()   # warm-up / compile
_ = g.count(lam[:2], 0.0); cp.cuda.Device().synchronize()
t0 = time.perf_counter()
g = sturm.GPUCounter(a2, n)
c_gpu = cp.asnumpy(g.count(lam, 0.0))
cp.cuda.Device().synchronize()
t_gpu = time.perf_counter() - t0

free, total = cp.cuda.runtime.memGetInfo()
lines = [
    f"device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  free/total memory {free/2**30:.1f}/{total/2**30:.1f} GiB",
    f"workload: n = {n} sites, {lam.size} values of lambda, one pass",
    f"CPU numpy (vectorised over lambda): {t_cpu:.2f} s",
    f"GPU cupy (incl. transfer + sync):  {t_gpu:.2f} s",
    f"speed-up: {t_cpu/t_gpu:.1f}x",
    f"counts identical: {bool(np.array_equal(c_cpu, c_gpu))}",
]
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
