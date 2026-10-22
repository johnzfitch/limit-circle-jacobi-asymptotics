"""Exact (SymPy) derivation of the second-order residual density of the
Langer-Airy comparison functions, used in the proof of the second-order
theorem (Section 9 of the paper).

Notation (paper, Section 9).  At a point t with Langer value zeta (> 0 on the
hyperbolic side), z = 1/(2 ahat), w = arccosh z, so that the eikonal identity
reads sqrt(zeta) zeta' = -w and the amplitude identity
g^4 = zeta / (ahat^2 sinh(w)^2).  The stencil solution y1(s; h) of
y'' = (zeta + h s) y, y(0) = 1, y'(0) = 0 is expanded as
X10 + h X11 + h^2 X12.  With F(h) = ahat(t+h/2) g(t+h) y1(s_+(h); h),
s_+(h) = (zeta(t+h) - zeta(t))/h, the residual coefficient is
P2 = F''(0) (P-tilde is even in h).  We verify:
  (1) X11, X12 solve their forced equations with zero data;
  (2) on the oscillatory side (zeta = -r, w = i theta),
      g P2 r^(-1/2) = D - (5/16) theta r^(-3),
      D = -[z'^2 (S^4 + 17 S^2 - 15) - z'' z (S^4 + 6 S^2)] / (12 S^5 z),
      S = sin(theta);
  (3) the eikonal term P0 vanishes;
  (4) the h^1 coefficient of Q-tilde (the discrete transport equation)
      vanishes, and the Casoratian factor ahat g^2 X2(zeta') equals -1
      (Lemma "Local residuals" of Section 7).
The check of (2) is exact: the difference is simplified to 0 after
substituting cos(theta) = z.
"""
import sympy as sp

t = sp.symbols('t', real=True)
s, h, k = sp.symbols('s h k', positive=True)

# ---- (1) stencil solutions --------------------------------------------------
X10 = sp.cosh(k * s)
X11 = s**2 * sp.sinh(k*s) / (4*k) - s * sp.cosh(k*s) / (4*k**2) + sp.sinh(k*s) / (4*k**3)
X12 = s * (3*k**3*s**3*sp.cosh(k*s) - 10*k**2*s**2*sp.sinh(k*s)
           + 21*k*s*sp.cosh(k*s) - 21*sp.sinh(k*s)) / (96*k**5)
res1 = sp.simplify(sp.diff(X11, s, 2) - k**2*X11 - s*X10)
res2 = sp.simplify(sp.diff(X12, s, 2) - k**2*X12 - s*X11)
data = [sp.simplify(X11.subs(s, 0)), sp.simplify(sp.diff(X11, s).subs(s, 0)),
        sp.simplify(sp.limit(X12, s, 0)), sp.simplify(sp.limit(sp.diff(X12, s), s, 0))]
print("(1) ODE residuals:", res1, res2, " initial data:", data)
assert res1 == 0 and res2 == 0 and all(d == 0 for d in data)

# ---- (2) the residual coefficient P2 -----------------------------------------
# Jets at the point: zeta, and z, z', z'', z''' (z''' cancels).  Work with
# w = arccosh z as the primary symbol: z = cosh w.
Z, w, z1, z2, z3 = sp.symbols('zeta w z1 z2 z3', positive=True)
z0 = sp.cosh(w)
# derivatives of w along t: w' = z'/sinh w, w'' = (z'' - cosh w w'^2)/sinh w, ...
W1 = z1 / sp.sinh(w)
W2 = (z2 - sp.cosh(w) * W1**2) / sp.sinh(w)
# zeta derivatives from sqrt(zeta) zeta' = -w  =>  (zeta^(3/2))' = -(3/2) w
T = sp.symbols('T')               # formal t
zeta_f = sp.Function('zeta')(T)
w_f = sp.Function('w')(T)
# zeta' = -w zeta^(-1/2)
d1 = -w_f * zeta_f**sp.Rational(-1, 2)
d2 = sp.diff(d1, T).subs(sp.Derivative(zeta_f, T), d1)
d3 = sp.diff(d2, T).subs(sp.Derivative(zeta_f, T), d1)
def sub_jets(e):
    e = e.subs(sp.Derivative(w_f, (T, 2)), W2).subs(sp.Derivative(w_f, T), W1)
    return e.subs(w_f, w).subs(zeta_f, Z)
Zp, Zpp, Zppp = [sub_jets(e) for e in (d1, d2, d3)]

# ahat = 1/(2z) and its derivatives
ah0 = 1 / (2*z0)
ah1 = -z1 / (2*z0**2)
ah2 = -z2 / (2*z0**2) + z1**2 / z0**3
# g = zeta^(1/4) (ahat sinh w)^(-1/2): derivatives by the chain rule along t
g_f = zeta_f**sp.Rational(1, 4) * (sp.Function('ah')(T) * sp.sinh(w_f))**sp.Rational(-1, 2)
g1 = sp.diff(g_f, T)
g2 = sp.diff(g1, T)
def sub_all(e):
    ahf = sp.Function('ah')(T)
    e = e.subs(sp.Derivative(ahf, (T, 2)), ah2).subs(sp.Derivative(ahf, T), ah1).subs(ahf, ah0)
    e = e.subs(sp.Derivative(zeta_f, (T, 2)), Zpp).subs(sp.Derivative(zeta_f, T), Zp)
    return sub_jets(e)
G0, G1, G2 = [sub_all(e) for e in (g_f, g1, g2)]

# F(h) to second order
kk = sp.sqrt(Z)
splus = Zp + h*Zpp/2 + h**2*Zppp/6
def y1(sv):
    return (X10 + h*X11 + h**2*X12).subs(k, kk).subs(s, sv)
F = (ah0 + (h/2)*ah1 + (h**2/8)*ah2) * (G0 + h*G1 + (h**2/2)*G2) * y1(splus)
F_ser = sp.series(F, h, 0, 3).removeO()
P0 = sp.simplify((2*F_ser.coeff(h, 0) - G0))
print("(3) eikonal term 2 F(0) - g:", P0)
P2 = 2 * F_ser.coeff(h, 2)            # P2 = F''(0) = 2 * [h^2] F
# --- continue to the oscillatory side: zeta = -r, sqrt(zeta) = i sqrt(r), w = i theta
r, th = sp.symbols('r theta', positive=True)
expr = G0 * P2
expr = expr.subs(sp.sqrt(Z), sp.I*sp.sqrt(r)).subs(Z, -r).subs(w, sp.I*th) / sp.sqrt(r)
expr = sp.expand(sp.simplify(expr.rewrite(sp.exp)))
zz = sp.cos(th)
S = sp.sin(th)
D = -(z1**2*(S**4 + 17*S**2 - 15) - z2*zz*(S**4 + 6*S**2)) / (12*S**5*zz)
target = D - sp.Rational(5, 16)*th/r**3
diff = sp.simplify(sp.expand((expr - target).rewrite(sp.exp)))
print("(2) g P2 r^(-1/2) - [D - (5/16) theta r^-3] =", diff)
# numeric spot check at random points as a second confirmation
import random
random.seed(1)
for _ in range(3):
    vals = {r: sp.Rational(random.randint(30, 300), 100), th: sp.Rational(random.randint(20, 130), 100),
            z1: sp.Rational(random.randint(-200, 200), 100), z2: sp.Rational(random.randint(-200, 200), 100)}
    print("   spot (exact rational point, 40 digits):", sp.N((expr - target).subs(vals), 40))

# ---- (4) the Q-residual and the Casoratian ----------------------------------
sig = sp.symbols('sigma')
X20 = sp.sinh(k*s) / k
X21 = sp.integrate(sp.sinh(k*(s - sig)) / k * sig * sp.sinh(k*sig) / k, (sig, 0, s))
X21 = sp.simplify(X21)
chk = sp.simplify(sp.diff(X21, s, 2) - k**2*X21 - s*X20)
print("(4a) X21 solves its forced equation:", chk, " data:",
      sp.simplify(X21.subs(s, 0)), sp.simplify(sp.diff(X21, s).subs(s, 0)))
def y2(sv):
    return (X20 + h*X21).subs(k, kk).subs(s, sv)
sminus = -Zp + h*Zpp/2 - h**2*Zppp/6
Qt = ((ah0 + (h/2)*ah1) * (G0 + h*G1) * y2(splus)
      + (ah0 - (h/2)*ah1) * (G0 - h*G1) * y2(sminus))
Q_ser = sp.series(Qt, h, 0, 2).removeO()
Q0 = sp.simplify(Q_ser.coeff(h, 0).rewrite(sp.exp))
Q1 = sp.simplify(sp.expand(Q_ser.coeff(h, 1).rewrite(sp.exp)))
print("(4b) h^0 coefficient of Q-tilde:", Q0)
print("(4c) h^1 coefficient of Q-tilde:", Q1)
cas = sp.simplify((ah0 * G0**2 * X20.subs(k, kk).subs(s, Zp)).rewrite(sp.exp))
print("(4d) Casoratian factor ahat g^2 X2(zeta'):", cas)
for _ in range(3):
    vals = {Z: sp.Rational(random.randint(10, 300), 100), w: sp.Rational(random.randint(10, 300), 100),
            z1: sp.Rational(random.randint(-200, 200), 100), z2: sp.Rational(random.randint(-200, 200), 100)}
    print("   spot Q1, cas+1:", sp.N(Q_ser.coeff(h, 1).subs(vals), 40), sp.N((ah0*G0**2*X20.subs(k, kk).subs(s, Zp)).subs(vals) + 1, 40))
