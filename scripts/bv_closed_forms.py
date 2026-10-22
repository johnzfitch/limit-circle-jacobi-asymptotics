"""Berg-Valent quartic problem in the normalisation of Berg-Szwarc (arXiv 2603.26302, (29)-(30)):
checks of the closed forms used in Section 'Application: critical indices' (CPU, mpmath, 50 digits).
G_F(x) = -cos(pi w/2) cosh(pi w/2), D(x) = (4d/pi^2) w^2 sin(pi w/2) sinh(pi w/2), w = x^(1/4), d = K0^2/pi.
1. masses 1/(D G_F') on odd n^4 and -1/(B D') on even n^4 (B = G_F there) against (29)-(30);
2. W_F = D/G_F' and W_inf = -B/D' against the closed forms;
3. the product law W_F W_inf Phi'(x) Phi'(y) Ups(y)/Ups(x) = tanh(pi n/2) coth(pi m/2) with
   Phi'(x) = (pi/8) x^(-3/4), Ups(x) proportional to x^(1/2);
4. the exponent of the series sum W_F(n^4) n^-8 n^-c (threshold c = -2)."""
import mpmath as mp
mp.mp.dps = 50
K0 = mp.gamma(mp.mpf(1)/4)**2 / (4*mp.sqrt(mp.pi)); d = K0**2/mp.pi
GF = lambda x: -mp.cos(mp.pi*x**0.25/2)*mp.cosh(mp.pi*x**0.25/2)
D = lambda x: 4*d/mp.pi**2 * mp.sqrt(x) * mp.sin(mp.pi*x**0.25/2)*mp.sinh(mp.pi*x**0.25/2)
worst = [0, 0, 0, 0]
for n in range(1, 30, 2):
    x = mp.mpf(n)**4
    m_closed = 4*mp.pi/K0**2 * n*mp.pi/mp.sinh(n*mp.pi)
    m = 1/(D(x)*mp.diff(GF, x))
    worst[0] = max(worst[0], abs(m/m_closed - 1))
    W = D(x)/mp.diff(GF, x)
    worst[1] = max(worst[1], abs(W/(32*d/mp.pi**3*n**5*mp.tanh(mp.pi*n/2)) - 1))
for n in range(2, 30, 2):
    y = mp.mpf(n)**4
    m_closed = 4*mp.pi/K0**2 * n*mp.pi/mp.sinh(n*mp.pi)
    m = -1/(GF(y)*mp.diff(D, y))
    worst[2] = max(worst[2], abs(m/m_closed - 1))
    W = -GF(y)/mp.diff(D, y)
    worst[3] = max(worst[3], abs(W/(2*mp.pi*n/d*mp.cosh(mp.pi*n/2)/mp.sinh(mp.pi*n/2)) - 1))
print("K0 =", mp.nstr(K0, 20), " d = K0^2/pi =", mp.nstr(d, 20), " mass at 0: 1/d =", mp.nstr(1/d, 15), "vs pi/K0^2 =", mp.nstr(mp.pi/K0**2, 15))
print("max rel. error, Friedrichs masses vs (29):", mp.nstr(worst[0], 5))
print("max rel. error, W_F closed form:            ", mp.nstr(worst[1], 5))
print("max rel. error, Krein masses vs (30):       ", mp.nstr(worst[2], 5))
print("max rel. error, W_inf closed form:          ", mp.nstr(worst[3], 5))
Php = lambda x: mp.pi/8 * x**(-mp.mpf(3)/4)
for k in (1, 3, 6, 12):
    n, mm = 2*k+1, 2*k+2
    x, y = mp.mpf(n)**4, mp.mpf(mm)**4
    WF = 32*d/mp.pi**3*n**5*mp.tanh(mp.pi*n/2); Wi = 2*mp.pi*mm/d/mp.tanh(mp.pi*mm/2)
    prod = WF*Wi*Php(x)*Php(y)*mp.sqrt(y/x)
    print(f"product law, n={n}, m={mm}: {mp.nstr(prod, 15)}   tanh*coth = {mp.nstr(mp.tanh(mp.pi*n/2)/mp.tanh(mp.pi*mm/2), 15)}")
for c in (-1, -2, -3):
    terms = [32*d/mp.pi**3*n**5*mp.tanh(mp.pi*n/2) * mp.mpf(n)**-8 * mp.mpf(n)**(-c) for n in (1001, 2001)]
    print(f"c = {c}: local exponent of the terms of sum W_F n^-8 n^-c = {mp.nstr(mp.log(terms[1]/terms[0])/mp.log(mp.mpf(2001)/1001), 8)} (convergent iff < -1)")
