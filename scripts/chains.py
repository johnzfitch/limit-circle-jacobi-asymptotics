"""Test chains for the numerical confirmations of the paper.

Every chain is a zero-diagonal Jacobi matrix in the paper's indexing:
(J u)_k = a_k u_{k-1} + a_{k+1} u_{k+1}, k >= 0, a_0 = 0, so a_k (k >= 1)
is the link between sites k-1 and k.  links(n) returns a_1, ..., a_n as a
float64 numpy array (index 0 holds a_1).

The parameter values were chosen afresh for this paper; none of them is a
chain used in the earlier research notes.
"""
import math
import numpy as np
from scipy.special import beta as Beta


def _power_curv(c, alpha, s, kappa2):
    """a_k = c (k+s)^alpha (1 + kappa2 (k+s)^-2)."""
    def links(n):
        k = np.arange(1, n + 1, dtype=np.float64) + s
        return c * np.exp(alpha * np.log(k)) * (1.0 + kappa2 / (k * k))
    return links


def _rough(c, alpha, ke, delta):
    """a_k = c k^alpha (1 + ke k^(-1-delta)): smooth symbol c x^alpha, size-only remainder."""
    def links(n):
        k = np.arange(1, n + 1, dtype=np.float64)
        return c * np.exp(alpha * np.log(k)) * (1.0 + ke * np.exp(-(1.0 + delta) * np.log(k)))
    return links


def _dimer(c, alpha, kappa, kappap):
    """a_k = c k^alpha (1 + (kappa + kappa' (-1)^k)/k)."""
    def links(n):
        k = np.arange(1, n + 1, dtype=np.float64)
        sgn = np.where(np.arange(1, n + 1) % 2 == 0, 1.0, -1.0)
        return c * np.exp(alpha * np.log(k)) * (1.0 + (kappa + kappap * sgn) / k)
    return links


def _squeeze(k_ph, m):
    """Block m of the k-photon squeezing operator a^k + a^{dagger k}:
    link between block sites n-1 and n is sqrt((m+(n-1)k+1)...(m+nk))."""
    def links(n):
        r = np.arange(0, n, dtype=np.float64)          # r = n-1
        lg = np.zeros(n)
        for l in range(1, k_ph + 1):
            lg += np.log(m + r * k_ph + l)
        return np.exp(0.5 * lg)
    return links


def _birth_death(p, e, d):
    """Symmetrised birth-death chain: a_{2n+1} = sqrt(lambda_n), a_{2n+2} = sqrt(mu_{n+1}),
    lambda_n = prod(p n + e_j), mu_n = prod(p n + d_j)."""
    def links(n):
        out = np.empty(n)
        idx = np.arange(1, n + 1)
        odd = idx % 2 == 1
        nn_odd = (idx[odd] - 1) // 2            # a_{2n+1}
        nn_even = idx[~odd] // 2                 # a_{2n+2} -> mu_{n+1}, n+1 = idx/2
        lo = np.zeros(nn_odd.size)
        for ej in e:
            lo += np.log(p * nn_odd + ej)
        le = np.zeros(nn_even.size)
        for dj in d:
            le += np.log(p * nn_even + dj)
        out[odd] = np.exp(0.5 * lo)
        out[~odd] = np.exp(0.5 * le)
        return out
    return links


def I_alpha(al):
    return 0.5 * Beta((al - 1) / (2 * al), 0.5)


def J_alpha(al):
    return Beta(0.5 + 1 / (2 * al), 0.5) / (2 * al)


CHAINS = {}

# Balanced chains with a curvature term (Theorem on the second-order constant).
for name, (c, al, s, k2) in {
    "B1": (0.7, 2.25, 0.35, 0.6),
    "B2": (1.3, 3.75, -0.2, -0.3),
    "B3": (0.5, 5.0, 0.6, 0.0),
}.items():
    CHAINS[name] = dict(kind="balanced", links=_power_curv(c, al, s, k2), c=c, alpha=al,
                        kappa=al * s, kappap=0.0, kappa2=k2,
                        second_order=(k2 + (2 * al - 1) / 24) * J_alpha(al))

# Squeezing blocks: c = k^(k/2), alpha = k/2, kappa = m/2 - (k-1)/4, kappa2 = -(k^2-1)/(48k).
for name, (kph, m) in {"SQ5m0": (5, 0), "SQ5m3": (5, 3), "SQ6m1": (6, 1)}.items():
    al = kph / 2
    kap = m / 2 - (kph - 1) / 4
    k2 = -(kph ** 2 - 1) / (48 * kph)
    CHAINS[name] = dict(kind="balanced", links=_squeeze(kph, m), c=kph ** (kph / 2), alpha=al,
                        kappa=kap, kappap=0.0, kappa2=k2,
                        second_order=(kph - 1) ** 2 / (48 * kph) * J_alpha(al))

# Size-only rough remainder (sharpness of the N^-delta term).
_al, _ke, _de = 3.0, 0.5, 0.5
CHAINS["RS"] = dict(kind="rough", links=_rough(1.0, _al, _ke, _de), c=1.0, alpha=_al, kappa=0.0,
                    kappap=0.0, delta=_de,
                    sh_const=_ke * Beta((_al + _de) / (2 * _al), 0.5) / (2 * _al))

# Dimerised chains (Bessel far field).
for name, (c, al, kap, kp) in {
    "D1": (1.0, 3.5, 0.2, 0.45),
    "D2": (0.8, 2.75, -0.3, -0.25),
}.items():
    CHAINS[name] = dict(kind="dimer", links=_dimer(c, al, kap, kp), c=c, alpha=al, kappa=kap, kappap=kp)

# Unbalanced polynomial birth-death rates, p = 6 (symmetric alpha = 3).
_p, _e, _d = 6, (2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 3, 3)
_E, _D = sum(_e), sum(_d)
CHAINS["BD6"] = dict(kind="dimer", links=_birth_death(_p, _e, _d), c=(_p / 2) ** (_p / 2), alpha=_p / 2,
                     kappa=(_D + _E) / (2 * _p) - _p / 4, kappap=(_D - _E) / (2 * _p) + _p / 4,
                     bd=dict(p=_p, E=_E, D=_D))


def Phi(ch, lam):
    """Phi(lambda) = N_lambda I_alpha - kappa pi / (2 alpha), N_lambda = (lambda/2c)^(1/alpha)."""
    al = ch["alpha"]
    N = (lam / (2 * ch["c"])) ** (1.0 / al)
    return N * I_alpha(al) - ch["kappa"] * math.pi / (2 * al), N
