"""Check of the q-family rows of Table 2 (Section 'Application: critical indices'), CPU, mpmath.

For an N-extremal solution mu_sigma with support Z_sigma, W_sigma(x) = 1/(mu_sigma({x}) G_sigma'(x)^2)
(Proposition 11.1 and eq. (Wsigma)), where G_sigma = B + sigma D (G_inf = D).  For the q-families the
Nevanlinna entries have order zero, so by Hadamard G_sigma = C z^m prod_j (1 - z/x_j) over the nonzero
support points; the constant C does not affect growth exponents.  We therefore compute, up to one
constant per solution,
        W(x_k) = 1 / ( m_k * G'(x_k)^2 ),   G(z) = z^m prod_{j} (1 - z/x_j),
from the masses m_k given in closed form (Al-Salam--Carlitz, Berg--Szwarc arXiv:2603.26302v1, Ex. 4.2)
or computed as m(x) = 1/sum_n p_n(x)^2 (Stieltjes--Wigert, Ex. 4.3), and fit the exponent of W in x.
Table 2 claims: Al-Salam--Carlitz W_F ~ x^(1+g), W_K ~ x^(1-g), g = log a/log(1/q);
Stieltjes--Wigert W_F ~ x^(3/2), W_K, W_1 ~ x^(1/2).  The index is then a = (1 -/+ exponent-shift)/2
and the critical series has terms W(x)(1+x^2)^(a-1) of bounded ratio, i.e. it diverges (class B)
as soon as W(x)/x^e stays bounded below.
Sanity check of the method first: Berg--Valent quartic, where W_F(n^4) = (32d/pi^3) n^5 tanh(pi n/2)."""
import mpmath as mp

mp.mp.dps = 60

def W_from_products(xs, masses, k, with_zero=False, J=None):
    """W at xs[k] (nonzero support points xs, ascending), G = z^[with_zero] prod (1 - z/x_j)."""
    x = xs[k]
    prod = mp.mpf(1)
    for j, xj in enumerate(xs if J is None else xs[:J]):
        if j != k:
            prod *= (1 - x/xj)
    # derivative of prod_j (1 - z/x_j) at its zero x_k is -(1/x_k) prod_{j != k}(1 - x_k/x_j)
    Gp = -prod/x
    if with_zero:
        Gp *= x          # G = z * prod  =>  G'(x_k) = x_k * prod'(x_k)
    return 1/(masses[k]*Gp**2)

def fit_exponents(xs, Ws, e_pred, label):
    print(f"  {label}: predicted exponent {mp.nstr(e_pred, 8)}")
    for k in range(2, len(xs) - 1, max(1, (len(xs) - 3)//6)):
        e_loc = mp.log(Ws[k+1]/Ws[k])/mp.log(xs[k+1]/xs[k])
        print(f"    x = {mp.nstr(xs[k], 6):>12}  local exponent {mp.nstr(e_loc, 10)}   W/x^e = {mp.nstr(Ws[k]/xs[k]**e_pred, 10)}")

# --- sanity: Berg--Valent quartic (Friedrichs support (2k+1)^4, masses (29)) ---
print("Sanity check, Berg--Valent quartic (W_F(x) ~ x^(5/4) tanh):")
K0 = mp.gamma(mp.mpf(1)/4)**2/(4*mp.sqrt(mp.pi))
NB = 400
xs = [mp.mpf(2*k+1)**4 for k in range(NB)]
ms = [4*mp.pi/K0**2*(2*k+1)*mp.pi/mp.sinh((2*k+1)*mp.pi) for k in range(NB)]
d = K0**2/mp.pi
for k in (3, 10, 30):
    W = W_from_products(xs, ms, k)
    n = 2*k+1
    print(f"    n = {n:3d}: W(products)/W(closed form) = {mp.nstr(W/(32*d/mp.pi**3*n**5*mp.tanh(mp.pi*n/2)), 12)}  (constant expected; truncation at {NB} factors)")

# --- Al-Salam--Carlitz, 0 < q < 1 < a < 1/q ---
def qpoch(z, q, n=None):
    if n is None:
        return mp.qp(z, q)
    return mp.qp(z, q, n)

for (q, a) in ((mp.mpf('0.5'), mp.mpf('1.5')), (mp.mpf('0.3'), mp.mpf('2.0'))):
    g = mp.log(a)/mp.log(1/q)
    print(f"\nAl-Salam--Carlitz, q = {q}, a = {a}, gamma = log a/log(1/q) = {mp.nstr(g, 10)}")
    K = 60
    xF = [a*q**(-k) - 1 for k in range(K)]
    mF = [qpoch(q/a, q)*a**(-k)*q**(k*k)/(qpoch(q/a, q, k)*qpoch(q, q, k)) for k in range(K)]
    WF = [W_from_products(xF, mF, k) for k in range(K - 20)]
    fit_exponents(xF[:K-20], WF, 1 + g, "Friedrichs W_F")
    # Krein: support 0 (k = 0) and q^-k - 1 (k >= 1); G = D = z prod_{k>=1}(1 - z/x_k)
    xK = [q**(-k) - 1 for k in range(1, K)]
    mK = [qpoch(a*q, q)*a**k*q**(k*k)/(qpoch(a*q, q, k)*qpoch(q, q, k)) for k in range(1, K)]
    WK = [W_from_products(xK, mK, k, with_zero=True) for k in range(K - 21)]
    fit_exponents(xK[:K-21], WK, 1 - g, "Krein W_K")

# --- Stieltjes--Wigert ---
def sw(q):
    mp.mp.dps = 260
    qq = q
    def Phi(x, kmax=400):
        s, t = mp.mpf(0), mp.mpf(1)
        poch = mp.mpf(1)
        for k in range(kmax):
            if k > 0:
                poch *= (1 - qq**k)
            term = (-1)**k*qq**(k*k)*x**k/poch
            s += term
            if k > 10 and abs(term) < abs(s)*mp.mpf(10)**(-250):
                break
        return s
    def dPhi(x):
        return mp.diff(Phi, x)
    # zeros of Phi: scan a logarithmic grid for sign changes
    zs = []
    grid = [mp.mpf(10)**(e/mp.mpf(40)) for e in range(-40, 40*14)]
    prev = Phi(grid[0])
    for i in range(1, len(grid)):
        cur = Phi(grid[i])
        if mp.sign(cur) != mp.sign(prev):
            zs.append(mp.findroot(Phi, (grid[i-1], grid[i]), solver='illinois', verify=False))
        prev = cur
    def mass(x):
        # p_n(x) = (-1)^n sqrt(q^n/(q;q)_n) sum_k [n k]_q (-1)^k q^(k^2) x^k   (Christiansen (2.4))
        N = int(3*mp.log(1 + abs(x))/mp.log(1/qq)) + 80
        tot = mp.mpf(0)
        qn = [mp.mpf(1)]
        for i in range(1, N + 2):
            qn.append(qn[-1]*(1 - qq**i))
        for n in range(N):
            s = mp.mpf(0)
            for k in range(n + 1):
                s += qn[n]/(qn[k]*qn[n-k])*(-1)**k*qq**(k*k)*x**k
            tot += qq**n/qn[n]*s**2
        return 1/tot
    print(f"\nStieltjes--Wigert, q = {mp.nstr(q, 3)}: {len(zs)} zeros of Ramanujan's function found, ratios xi_(n+1)/xi_n:",
          [mp.nstr(zs[i+1]/zs[i], 6) for i in range(min(5, len(zs)-1))])
    xs = zs
    WF = [1/(mass(x)*dPhi(x)**2) for x in xs]
    W1 = [1/(mass(qq*x)*(dPhi(x)/qq)**2) for x in xs]           # G_1(z) = Phi(z/q)
    WK = [1/(mass(x/qq)*(x*dPhi(x))**2) for x in xs]           # D(z) = z Phi(q z), D'(xi/q) = xi Phi'(xi)
    fit_exponents(xs, WF, mp.mpf(3)/2, "Friedrichs W_F (support xi_n)")
    fit_exponents([qq*x for x in xs], W1, mp.mpf(1)/2, "sigma = 1, W_1 (support q xi_n)")
    fit_exponents([x/qq for x in xs], WK, mp.mpf(1)/2, "Krein W_K (support xi_n/q)")
    mp.mp.dps = 60

sw(mp.mpf('0.5'))
sw(mp.mpf('0.3'))
