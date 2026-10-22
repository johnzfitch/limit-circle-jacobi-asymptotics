"""Check of the closed form of varpi for symmetrised birth-death chains (Section 6):
varpi = lim A_m P_{2m}^2 N_m^{2 kappa'} = p^{p/2} 2^{2 kappa'} prod Gamma(1 + d_j/p) / Gamma(e_j/p),
on the chain BD6 (p = 6), by direct products up to m = 2e6 (CPU, float64 log-sums)."""
import numpy as np, math, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from chains import CHAINS
from scipy.special import gammaln
ch = CHAINS['BD6']; p = 6; e = (2, 3, 4, 5, 6, 7); d = (0, 1, 2, 3, 3, 3); kp = ch['kappap']
a = ch['links'](4000002); la = np.log(a)
f = (p / 2) * math.log(p) + 2 * kp * math.log(2) + sum(gammaln(1 + dj / p) for dj in d) - sum(gammaln(ej / p) for ej in e)
print(f"closed form varpi = {math.exp(f):.12f}")
for M in (1000, 100000, 2000000):
    logP2 = np.sum(2 * (la[0:2 * M:2] - la[1:2 * M:2]))
    L = la[2 * M] + logP2 + 2 * kp * math.log(2 * M + 1)
    print(f"m = {M:8d}: A_m P_2m^2 N_m^(2 kappa') = {math.exp(L):.12f}   relative difference {math.exp(L - f) - 1:+.2e}")
