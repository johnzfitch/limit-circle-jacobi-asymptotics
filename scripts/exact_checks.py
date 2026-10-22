"""Small exact and ball-arithmetic checks of the constants used in the paper.

SymPy is used for exact identities (symbolic in alpha where possible);
python-flint (arb ball arithmetic) and mpmath for numerical values at chosen
exponents.  Every check prints PASS/FAIL; the script exits non-zero on FAIL.
"""
import sys
import sympy as sp
import mpmath as mp
import flint
from flint import arb, acb

mp.mp.dps = 40
flint.ctx.prec = 200
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fails.append(name)


a = sp.symbols('alpha', positive=True)


def sfac(u, al):
    """2u / sqrt(1 - (1+u^2)^(-2 al)), continuous at u = 0 (t = 1 + u^2 substitution)."""
    if u == 0:
        return 2 / mp.sqrt(2*al)
    return 2*u / mp.sqrt(-mp.expm1(-2*al*mp.log1p(u*u)))
x = sp.symbols('x', positive=True)

# --- C1: I_alpha = int_0^oo arcsin(min(1, t^-alpha)) dt = (1/2) B((alpha-1)/(2alpha), 1/2)
for al in (sp.Rational(9, 8), sp.Rational(9, 4), 3, sp.Rational(15, 4), 5):
    alf = mp.mpf(al.p)/al.q if isinstance(al, sp.Rational) else mp.mpf(int(al))
    lhs = mp.pi/2 + mp.quad(lambda u: mp.asin(mp.exp(-alf*u)) * mp.exp(u), [0, 1, 10, mp.inf])
    rhs = mp.beta((alf - 1)/(2*alf), mp.mpf(1)/2) / 2
    check(f"C1 I_alpha integral = Beta form, alpha={al}", abs(lhs - rhs) < mp.mpf(10)**-30, mp.nstr(lhs - rhs, 3))

# --- C2: C_alpha = sin(pi/2a) B(1/(2a), 1 - 1/a)/(2 pi) = 2^(-1/a) I_alpha / pi (duplication), symbolic
Ia = sp.gamma((a - 1)/(2*a)) * sp.gamma(sp.Rational(1, 2)) / sp.gamma(1 - 1/(2*a)) / 2
Ca = sp.sin(sp.pi/(2*a)) * sp.gamma(1/(2*a)) * sp.gamma(1 - 1/a) / sp.gamma(1 - 1/(2*a)) / (2*sp.pi)
for al in (sp.Rational(9, 8), sp.Rational(5, 2), 7):
    v = sp.N((Ca - 2**(-1/a)*Ia/sp.pi).subs(a, al), 40)
    check(f"C2 counting constant duplication, alpha={al}", abs(v) < 1e-35, str(v))

# --- C3: drift integral int_1^oo t^-a (1 - t^-2a)^-1/2 dt/t = pi/(2 a)
for al in (1.5, 2.75, 3.5, 6.0):
    al = mp.mpf(al)
    v = mp.quad(lambda u: sfac(u, al) * (1+u*u)**(-al-1), [0, 1, mp.inf])
    check(f"C3 drift integral = pi/(2 alpha), alpha={al}", abs(v - mp.pi/(2*al)) < mp.mpf(10)**-25, mp.nstr(v - mp.pi/(2*al), 3))

# --- C4: J_alpha = int_1^oo t^(-a-2) (1 - t^-2a)^-1/2 dt = B(1/2 + 1/(2a), 1/2)/(2a)
for al in (2.25, 3.75, 5.0):
    al = mp.mpf(al)
    v = mp.quad(lambda u: sfac(u, al) * (1+u*u)**(-al-2), [0, 1, mp.inf])
    r = mp.beta(mp.mpf(1)/2 + 1/(2*mp.mpf(al)), mp.mpf(1)/2) / (2*al)
    check(f"C4 J_alpha, alpha={al}", abs(v - r) < mp.mpf(10)**-25, mp.nstr(v - r, 3))

# --- C5: finite-part evaluation f.p. int D dt = (2a-1) J_a / 12 (exact in alpha)
# M(gamma) := (1/2) B(p, 1 - gamma), p = 1/2 + 1/(2a), continued analytically in gamma
p = sp.Rational(1, 2) + 1/(2*a)
J = sp.gamma(p)*sp.gamma(sp.Rational(1, 2))/sp.gamma(p + sp.Rational(1, 2)) / (2*a)
def M(g):
    return sp.gamma(p)*sp.gamma(1 - g)/sp.gamma(p + 1 - g) / 2
fp = -sp.Rational(1, 12) * (-M(sp.Rational(1, 2)) + (11*a - 6)*M(sp.Rational(3, 2)) - 15*a*M(sp.Rational(5, 2)))
ratio = sp.simplify(sp.gammasimp(fp / J))
check("C5 f.p. int D dt = (2 alpha - 1) J_alpha / 12 (symbolic)", sp.simplify(ratio - (2*a - 1)/12) == 0, str(ratio))

# --- C6: the w-substitution of D: D dt = -(1/12) w^(1/a) [-s^-1 + (11a-6) s^-3 - 15 a s^-5] dw
w = sp.symbols('w', positive=True)
tt = w**(-1/a)
z, z1, z2 = w, sp.diff(w**1, w), 0     # placeholders
# z = t^-a ; z' = -a t^(-a-1) ; z'' = a(a+1) t^(-a-2)
zt = tt**(-a); zp = -a*tt**(-a-1); zpp = a*(a+1)*tt**(-a-2)
S = sp.sqrt(1 - zt**2)
D = -(zp**2*(S**4 + 17*S**2 - 15) - zpp*zt*(S**4 + 6*S**2)) / (12*S**5*zt)
dtdw = sp.diff(tt, w)
sig = sp.sqrt(1 - w**2)
target = sp.Rational(1, 12) * w**(1/a) * (-1/sig + (11*a - 6)/sig**3 - 15*a/sig**5)   # dt/dw < 0: int_1^oo D dt = -(1/12) int_0^1 w^(1/a)[...] dw
diffw = sp.simplify(sp.powsimp(sp.expand(D*dtdw - target), force=True))
num_ok = True
for al in (1.7, 3.2):
    for wv in (0.2, 0.55, 0.9):
        dv = complex((D*dtdw - target).subs({a: al, w: wv}).evalf(30))
        num_ok &= abs(dv) < 1e-20
check("C6 w-substitution of the local density D (exact)", diffw == 0, f"symbolic remainder: {diffw}")

# --- C7: K2 = (1/2) int_1^oo G2 dt by direct quadrature, G2 = D - (5/16) theta r^-3
mp.mp.dps = 60
def K2_quad(al):
    al = mp.mpf(al)
    def theta(t): return mp.acos(t**(-al))
    def A(t): return mp.quad(theta, [1, t])
    def G2(t):
        zt = t**(-al); zp = -al*t**(-al-1); zpp = al*(al+1)*t**(-al-2)
        Sv = mp.sqrt(1 - zt**2)
        Dv = -(zp**2*(Sv**4 + 17*Sv**2 - 15) - zpp*zt*(Sv**4 + 6*Sv**2)) / (12*Sv**5*zt)
        rv = (mp.mpf(3)/2*A(t))**(mp.mpf(2)/3)
        return Dv - mp.mpf(5)/16*theta(t)/rv**3
    # split: [1, 2] with t = 1 + v^2, tail [2, oo) with t^-2 decay
    # near t = 1 the two terms of G2 cancel to (t-1)^(-1/2): Gauss-Legendre nodes avoid the endpoint
    I1 = mp.quad(lambda v: G2(1 + v*v)*2*v, [mp.mpf(10)**-12, mp.mpf(10)**-4, 0.01, 0.25, 1], method='gauss-legendre')
    I2 = mp.quad(G2, [2, 8, 64, mp.inf])
    return (I1 + I2)/2
for al in (2.25, 3.75):
    K = K2_quad(al)
    pred = (2*al - 1)/24 * mp.beta(mp.mpf(1)/2 + 1/(2*mp.mpf(al)), mp.mpf(1)/2)/(2*al)
    check(f"C7 K2 by quadrature = (2a-1)J/24, alpha={al}", abs(K/pred - 1) < mp.mpf(10)**-9, mp.nstr(K/pred - 1, 3))
mp.mp.dps = 40

# --- C8: Weyl-Moyal cross-check (an independent, formal route to the constant).
# For A = X^a (c = 1), z = E/(2A), X = X_E w^(-1/a): curvature part I(E) = int -2 A'' z^2 (1-z^2)^(-1/2) dX,
# gradient part dII/dE = (2/E) int (A'^2/A) z^2 (1-z^2)^(-1/2) dX.  Both reduce to M(1/2) = alpha J.
XE, E_ = sp.symbols('X_E E', positive=True)
Xw = XE*w**(-1/a)
dXdw = sp.diff(Xw, w)                         # negative; int_{X_E}^oo dX = int_1^0 (...) dw
Aw = Xw**a; A1 = a*Xw**(a - 1); A2 = a*(a - 1)*Xw**(a - 2)
zw = w                                        # z = (X_E/X)^a = w
intI = sp.simplify(sp.powsimp(-2*A2*zw**2/sp.sqrt(1 - zw**2)*(-dXdw), force=True))
intII = sp.simplify(sp.powsimp((A1**2/Aw)*zw**2/sp.sqrt(1 - zw**2)*(-dXdw), force=True))
kerI = sp.simplify(intI/(w**(1/a)/sp.sqrt(1 - w**2)))
kerII = sp.simplify(intII/(w**(1/a)/sp.sqrt(1 - w**2)))
check("C8a curvature integrand = -2(alpha-1) X_E^(alpha-1) w^(1/alpha)(1-w^2)^(-1/2)", sp.simplify(kerI + 2*(a - 1)*XE**(a - 1)) == 0, str(kerI))
check("C8b gradient integrand = alpha X_E^(alpha-1) w^(1/alpha)(1-w^2)^(-1/2)", sp.simplify(kerII - a*XE**(a - 1)) == 0, str(kerII))
Mh = M(sp.Rational(1, 2))
XEofE = (E_/2)**(1/a)
I_E = -2*(a - 1)*XEofE**(a - 1)*Mh
CI = sp.simplify(-sp.diff(I_E, E_)/24)
dII = (2/E_)*a*XEofE**(a - 1)*Mh
CII = sp.simplify(-dII/24)
NE = XEofE
check("C8c N*C_I = (alpha-1)^2 J/24", sp.simplify(sp.gammasimp(NE*CI - (a - 1)**2*J/24)) == 0)
check("C8d N*C_II = -alpha^2 J/24", sp.simplify(sp.gammasimp(NE*CII + a**2*J/24)) == 0)
check("C8e N*(C_I + C_II) = -(2alpha-1) J/24", sp.simplify(sp.gammasimp(NE*(CI + CII) + (2*a - 1)*J/24)) == 0)

# --- C9: Bessel far field constants
nu, tau = sp.symbols('nu tau', positive=True)
Wr = sp.simplify(sp.besselj(-nu, tau)*sp.besselj(nu - 1, tau) + sp.besselj(1 - nu, tau)*sp.besselj(nu, tau))
for nv in (0.2, 0.37, 0.8):
    for tv in (0.3, 2.0, 7.5):
        val = mp.besselj(-nv, tv)*mp.besselj(nv - 1, tv) + mp.besselj(1 - nv, tv)*mp.besselj(nv, tv)
        ok = abs(tv*val - 2*mp.sin(nv*mp.pi)/mp.pi) < mp.mpf(10)**-30
        if not ok:
            break
check("C9 Wronskian tau(J_-nu J_{nu-1} + J_{1-nu} J_nu) = (2/pi) sin(nu pi)", ok)
# det M0 with a = T - pi/4 + nu pi/2, b = T - pi/4 - nu pi/2: det [[cos a, cos b],[-sin a, -sin b]] = sin(nu pi)
T_, nu_ = sp.symbols('T nu', real=True)
aa = T_ - sp.pi/4 + nu_*sp.pi/2; bb = T_ - sp.pi/4 - nu_*sp.pi/2
detM0 = sp.simplify(sp.Matrix([[sp.cos(aa), sp.cos(bb)], [-sp.sin(aa), -sp.sin(bb)]]).det())
check("C10 det M0 = +sin(nu pi)", sp.simplify(detM0 - sp.sin(nu_*sp.pi)) == 0, str(detM0))
# solve M0 c = (cos chi, sin chi)
chi = sp.symbols('chi', real=True)
sol = sp.Matrix([[sp.cos(aa), sp.cos(bb)], [-sp.sin(aa), -sp.sin(bb)]]).LUsolve(sp.Matrix([sp.cos(chi), sp.sin(chi)]))
c1 = sp.simplify(sol[0] + sp.sin(chi + bb)/sp.sin(nu_*sp.pi)); c2 = sp.simplify(sol[1] - sp.sin(chi + aa)/sp.sin(nu_*sp.pi))
check("C11 c_- = -sin(chi+b)/sin(nu pi), c_+ = sin(chi+a)/sin(nu pi)",
      sp.simplify(sp.expand_trig(c1)) == 0 and sp.simplify(sp.expand_trig(c2)) == 0, f"{c1}, {c2}")
# small-tau limits: Q = (2(alpha-1)/Lambda)^(1-2nu) Gamma(1-nu)/Gamma(nu)
for (al, kp, Lam) in ((3.5, 0.45, 1e5), (2.75, -0.25, 3e4)):
    al, kp, Lam = mp.mpf(al), mp.mpf(kp), mp.mpf(Lam)
    nv = mp.mpf(1)/2 - kp/(al - 1)
    xi = (Lam/(al - 1))**(-kp/(al - 1))
    d1 = 2**nv/(mp.gamma(1 - nv)*xi); d2 = xi*2**(1 - nv)/mp.gamma(nv)
    Q = (2*(al - 1)/Lam)**(1 - 2*nv)*mp.gamma(1 - nv)/mp.gamma(nv)
    # the limits themselves, from the series of J_{-nu}, J_{nu-1} at large N
    Nn = mp.mpf(10)**60
    tv = Lam*Nn**(1 - al)/(al - 1)
    y1 = Nn**kp*mp.sqrt(tv)*mp.besselj(-nv, tv); y2 = Nn**(-kp)*mp.sqrt(tv)*mp.besselj(nv - 1, tv)
    check(f"C12 Bessel limits and Q, alpha={al}, kappa'={kp}",
          abs(y1/d1 - 1) < 1e-10 and abs(y2/d2 - 1) < 1e-10 and abs(d2/d1/Q - 1) < 1e-25)

# --- C13: Hurwitz zeta range on (0,1]: min at theta = 1, max 0.0933663909422 at 0.3027218...
f = lambda th: mp.zeta(-0.5, th)
thmax = mp.findroot(lambda th: mp.diff(f, th), 0.3)
check("C13 max of zeta(-1/2, theta)", abs(f(thmax) - mp.mpf("0.0933663909422")) < 1e-12 and abs(thmax - mp.mpf("0.302721828598")) < 1e-11,
      f"theta*={mp.nstr(thmax, 13)} max={mp.nstr(f(thmax), 13)} min={mp.nstr(f(1), 13)}")

# --- C14: squeezing blocks: a_n = sqrt(prod_{l=1}^k (m+(n-1)k+l)) = k^(k/2) (n+s)^(k/2) (1 + kappa2 (n+s)^-2 + O(n^-3))
nn = sp.symbols('n', positive=True)
for kph in (3, 4, 5, 6):
    for m in range(kph):
        s_ = sp.Rational(2*m - kph + 1, 2*kph)
        k2 = -sp.Rational(kph**2 - 1, 48*kph)
        expr = sp.log(sp.Mul(*[(m + (nn - 1)*kph + l) for l in range(1, kph + 1)]))/2 \
            - (sp.Rational(kph, 2)*sp.log(kph) + sp.Rational(kph, 2)*sp.log(nn + s_) + sp.log(1 + k2/(nn + s_)**2))
        ser = sp.series(expr.subs(nn, 1/x), x, 0, 3).removeO()
        if sp.simplify(ser) != 0:
            check(f"C14 squeezing expansion k={kph} m={m}", False, str(ser)); break
    else:
        continue
    break
else:
    check("C14 squeezing blocks: shift (2m-k+1)/(2k), kappa2 = -(k^2-1)/(48k), k = 3..6", True)
kk = sp.symbols('k', positive=True)
const = sp.simplify(-(kk**2 - 1)/(48*kk) + (kk - 1)/24)
check("C15 squeezing second-order coefficient kappa2 + (2alpha-1)/24 = (k-1)^2/(48k)", sp.simplify(const - (kk - 1)**2/(48*kk)) == 0)

# --- C16: birth-death dictionary (symmetrised chain) and the Krein/Friedrichs offsets
pp = sp.symbols('p', positive=True)
Es, Ds = sp.symbols('E D', real=True)
kap = (Ds + Es)/(2*pp) - pp/4; kapp = (Ds - Es)/(2*pp) + pp/4
nu_bd = sp.Rational(1, 2) - kapp/(pp/2 - 1)
check("C16a nu = ((E-D)/p - 1)/(p-2)", sp.simplify(nu_bd - ((Es - Ds)/pp - 1)/(pp - 2)) == 0)
theta0 = kapp*sp.pi/(2*(pp/2 - 1))
krein = sp.simplify((theta0 + kap*sp.pi/pp)/sp.pi - ((Ds + Es)/(2*pp**2) - nu_bd/2))
check("C16b Krein offset (D+E)/(2p^2) - nu/2", krein == 0, str(krein))
# odd links see kappa - kappa' = E/p - p/2, even links kappa + kappa' = D/p
check("C16c odd/even link constants", sp.simplify(kap - kapp - (Es/pp - pp/2)) == 0 and sp.simplify(kap + kapp - Ds/pp) == 0)

# --- C17: Valent density identity: tau_p sin(pi/p)/pi = C_alpha c^(-1/alpha) with alpha = p/2, c = (p/2)^(p/2)
for pv in (3, 4, 5, 6, 8, 10):
    tau = mp.beta(1/mp.mpf(pv), 1 - 2/mp.mpf(pv))/pv
    al = mp.mpf(pv)/2; c = al**al
    Cal = mp.sin(mp.pi/(2*al))*mp.beta(1/(2*al), 1 - 1/al)/(2*mp.pi)
    ok = abs(tau*mp.sin(mp.pi/pv)/mp.pi - Cal*c**(-1/al)) < mp.mpf(10)**-35
    if not ok: break
check("C17 Valent density = C_alpha c^(-1/alpha) for balanced rates, p = 3..10", ok)

# --- C18: arb (ball arithmetic) evaluation of the headline constants at the test exponents
for al_s in ("2.25", "3.75", "5"):
    al = arb(al_s)
    Ib = arb.beta((al - 1)/(2*al), arb(0.5))/2 if hasattr(arb, "beta") else None
    Jb = (arb(0.5) + 1/(2*al)).gamma()*arb(0.5).gamma()/((arb(1) + 1/(2*al)).gamma())/(2*al)
    print(f"      alpha={al_s}: J_alpha = {Jb.str(20)}  (2alpha-1)J/24 = {((2*al - 1)*Jb/24).str(20)}")
check("C18 arb evaluation completed", True)


# --- C19: block rotation  E~^-1 T1 E~ = R(phi), phi = 2 arcsin(mu/2), E~ e+ = e^-s e+, E~ e- = e^s e-
mu_ = sp.symbols('mu', positive=True)
sv = sp.log((2 + mu_)/(2 - mu_))/4
ep = sp.Matrix([1, 1])/sp.sqrt(2); em = sp.Matrix([1, -1])/sp.sqrt(2)
Et = sp.exp(-sv)*ep*ep.T + sp.exp(sv)*em*em.T
T1 = sp.Matrix([[1 - mu_**2, -mu_], [mu_, 1]])
phi_ = 2*sp.asin(mu_/2)
Rphi = sp.Matrix([[sp.cos(phi_), -sp.sin(phi_)], [sp.sin(phi_), sp.cos(phi_)]])
Dm = sp.simplify(Et.inv()*T1*Et - Rphi)
okb = all(sp.simplify(sp.radsimp(e)) == 0 for e in Dm) or all(abs(complex(e.subs(mu_, v).evalf(60))) < 1e-40 for e in Dm for v in (sp.Rational(3, 10), sp.Rational(11, 10), sp.Rational(19, 10)))
check("C19 block rotation E~^-1 T1(mu) E~ = R(2 arcsin(mu/2))", okb)
# window matrix: C^-1 (-T1(2-2d)) C with C = [e_-, e_+/(4 l)]
dd, ll = sp.symbols('d l', positive=True)
Cm = sp.Matrix.hstack(em, ep/(4*ll))
Mw = sp.simplify(Cm.inv()*(-T1.subs(mu_, 2 - 2*dd))*Cm)
target_w = sp.Matrix([[1 - 4*dd + 2*dd**2, (1 - dd)*(2 - dd)/(2*ll)], [-8*ll*dd*(1 - dd), 1 - 4*dd + 2*dd**2]])
check("C20 Airy-window block matrix", sp.simplify(Mw - target_w) == sp.zeros(2, 2), str(sp.simplify(Mw)))
# frame-change generator: E~^-1 Sigma3 E~ = cosh(2s) Sigma3 + sinh(2s) J
S3 = sp.diag(1, -1); Jm = sp.Matrix([[0, -1], [1, 0]])
X = Et.inv()*S3*Et - (sp.cosh(2*sv)*S3 + sp.sinh(2*sv)*Jm)   # not simplified: sympy hangs on it
# symbolic simplification of X is slow (it hung for over an hour); check the entries in a
# variable q = e^{s} > 0 instead, where the identity is rational in q, plus 60-digit spot values.
q = sp.symbols('q', positive=True)
Etq = q**-1*ep*ep.T + q*em*em.T
Xq = Etq.inv()*S3*Etq - (((q**2 + q**-2)/2)*S3 + ((q**2 - q**-2)/2)*Jm)
okq = all(sp.simplify(e) == 0 for e in Xq)
okn = all(abs(complex(e.subs(mu_, v).evalf(60))) < 1e-40 for e in X for v in (sp.Rational(3, 10), sp.Rational(19, 10)))
check("C21 frame-change generator", okq and okn)

print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
