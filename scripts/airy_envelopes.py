"""Facts (A1)-(A4) about Olver's Airy envelopes used in Theorem D (Section 7).

E(x) = sqrt(Bi/Ai), M(x) = sqrt(2 Ai Bi) for x >= c*,  E = 1, M = sqrt(Ai^2+Bi^2)
for x <= c*, where c* is the largest root of Ai = Bi.  On a grid we check:
(A1) E nondecreasing; (A2) |Ai| <= M/E, |Bi| <= E M;
(A3) m0 = sup M^2 (1+|x|)^{1/2}; (A4) k0 = sup k(x) (1+|x|)^{-1/2},
k(x) = max(|Ai'|/(M/E), |Bi'|/(E M)).  mpmath at 30 digits.
"""
import mpmath as mp
mp.mp.dps = 30
cstar = mp.findroot(lambda x: mp.airyai(x) - mp.airybi(x), -0.37)

def env(x):
    ai, bi = mp.airyai(x), mp.airybi(x)
    if x >= cstar:
        return mp.sqrt(bi / ai), mp.sqrt(2 * ai * bi)
    return mp.mpf(1), mp.sqrt(ai**2 + bi**2)

xs = [mp.mpf(-400) + mp.mpf(i) / 50 for i in range(0, 50 * 440 + 1)]   # [-400, 40]
xs += [mp.mpf(-1e4), mp.mpf(-1e6), mp.mpf(100)]
xs.sort()
Eprev = None
mono = True
a2 = 0
m0 = (0, None)
k0 = (0, None)
for x in xs:
    E, M = env(x)
    ai, bi = mp.airyai(x), mp.airybi(x)
    aip, bip = mp.airyai(x, 1), mp.airybi(x, 1)
    if Eprev is not None and E < Eprev * (1 - mp.mpf(10)**-25):
        mono = False
    Eprev = E
    a2 = max(a2, abs(ai) / (M / E), abs(bi) / (E * M))
    val = M**2 * mp.sqrt(1 + abs(x))
    if val > m0[0]:
        m0 = (val, x)
    k = max(abs(aip) / (M / E), abs(bip) / (E * M))
    val = k / mp.sqrt(1 + abs(x))
    if val > k0[0]:
        k0 = (val, x)
print("c* =", mp.nstr(cstar, 15))
print("(A1) E nondecreasing on grid:", mono, " E(c*) =", mp.nstr(env(cstar)[0], 15))
print("(A2) max(|Ai|/(M/E), |Bi|/(EM)) =", mp.nstr(a2, 15))
print("(A3) m0 = sup M^2 (1+|x|)^(1/2) =", mp.nstr(m0[0], 8), "at x =", mp.nstr(m0[1], 6))
print("     M^2 |x|^(1/2) at x=-1e6:", mp.nstr(env(mp.mpf(-1e6))[1]**2 * 1000, 12), " 1/pi =", mp.nstr(1/mp.pi, 12))
print("(A4) sup k (1+|x|)^(-1/2) =", mp.nstr(k0[0], 8), "at x =", mp.nstr(k0[1], 6))
print("grid points:", len(xs))
