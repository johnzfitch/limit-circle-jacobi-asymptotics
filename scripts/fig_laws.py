"""Figure 2 of the paper: the eigenvalue laws of H_0, H_{pi/2} and a generic
extension, on a balanced and on a dimerised chain.

Reads only the extrapolated truncation data already in outputs/trunc_cases
(produced by truncation_run.py); no new eigenvalues are computed.  For each
level the plotted value is r_extrap = Phi(lambda_j) - j*pi.

Usage:  python scripts/fig_laws.py            (writes outputs/fig_laws.pdf)
The kit's es_figure.py must be importable (PYTHONPATH=<kit>/python).
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CASES = os.path.join(ROOT, "outputs", "trunc_cases")
OUT = os.path.join(ROOT, "outputs", "fig_laws.pdf")

from es_figure import es_style, es_save, ES_PALETTE, ES_GREY  # noqa: E402

backend = "pgf" if "--mathtext" not in sys.argv else None
if backend:
    es_style(backend="pgf")
else:
    es_style()
import matplotlib.pyplot as plt  # noqa: E402


def load(name):
    with open(os.path.join(CASES, name + ".json")) as f:
        d = json.load(f)
    xs = [row["N"] for row in d["rows"]]
    ys = [row["r_extrap"] for row in d["rows"]]
    return xs, ys


alpha_d1, kappap_d1 = 3.5, 0.45
theta0 = kappap_d1 * math.pi / (2 * (alpha_d1 - 1))

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), sharey=True)
series = [("t0.000000", r"$t=0$", ES_PALETTE[0], "o", 3.6, "full"),
          ("t1.000000", r"$t=1$", ES_PALETTE[1], "s", 6.0, "none"),
          ("t1.570796", r"$t=\pi/2$", ES_PALETTE[2], "^", 3.6, "full")]

ax = axes[0]
for tag, lab, col, mk, ms, fill in series:
    xs, ys = load("B3_" + tag)
    ax.plot(xs, ys, marker=mk, ms=ms, fillstyle=fill, mew=1.0, lw=0.9, color=col, label=lab)
for y in (0.0, 1.0, math.pi / 2):
    ax.axhline(y, color=ES_GREY, lw=0.6, ls="--", zorder=0)
ax.set_title("balanced (B3)", fontsize=10.5)

ax = axes[1]
for tag, lab, col, mk, ms, fill in series:
    xs, ys = load("D1_" + tag)
    ls = "--" if tag == "t1.000000" else "-"
    ax.plot(xs, ys, marker=mk, ms=ms, fillstyle=fill, mew=1.0, lw=0.9, ls=ls, color=col, label=lab,
            zorder=3 if tag == "t1.000000" else 2)
for y in (theta0, math.pi / 2 - theta0):
    ax.axhline(y, color=ES_GREY, lw=0.6, ls="--", zorder=0)
ax.axhline(1.0, color=ES_GREY, lw=0.6, ls=":", zorder=0)
ax.text(330, theta0 + 0.04, r"$\theta_0$", fontsize=9, ha="right", va="bottom")
ax.text(330, math.pi / 2 - theta0 + 0.04, r"$\pi/2-\theta_0$", fontsize=9, ha="right", va="bottom")
ax.text(330, 1.0 - 0.05, r"balanced value $1$", fontsize=8.5, ha="right", va="top", color=ES_GREY)
ax.set_title(r"dimerised (D1, $\kappa'>0$)", fontsize=10.5)

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(8, 400)
    ax.set_xlabel(r"$N_{\lambda_j}$", fontsize=10.5)
    ax.tick_params(labelsize=9)
axes[0].set_ylabel(r"$\Phi(\lambda_j(t))-j\pi$", fontsize=10.5)
axes[0].set_ylim(-0.12, 1.75)
axes[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.31), fontsize=9, frameon=True)
fig.tight_layout()
es_save(fig, OUT)
print("wrote", OUT)
print("theta0(D1) =", theta0, " pi/2 - theta0 =", math.pi / 2 - theta0)
