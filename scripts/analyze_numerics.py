"""Tabulate the truncation results against the paper's predictions (Section 11).

For every chain and extension t, r_j := Phi(lambda_j) - j pi - off_t, where Phi is the
first-order phase, lambda_j the extrapolated eigenvalue of the truncations, and off_t
the predicted offset (t for balanced chains, theta_0 or pi/2 - theta_0 for dimerised
chains).  Printed: the scaled residual at the largest levels and its prediction.
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from chains import CHAINS, J_alpha
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
import glob
res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "trunc_cases", "*.json")))]
lines = []
def say(s):
    print(s); lines.append(s)
for r in res:
    ch = CHAINS[r["chain"]]; t = r["t"]; al = ch["alpha"]; kp = ch.get("kappap", 0.0)
    if ch["kind"] == "dimer":
        th0 = kp * math.pi / (2 * (al - 1))
        if t == 0.0: off = th0
        elif abs(t - math.pi / 2) < 1e-12: off = math.pi / 2 - th0
        else: off = math.pi / 2 - th0 if kp > 0 else th0 + (math.pi if t > math.pi / 2 else 0.0)
    else:
        off = t
    rows = r["rows"]
    say(f"== {r['chain']}  t = {t:.6f}  ({ch['kind']}, alpha = {al}, kappa = {ch['kappa']:.6g}, kappa' = {kp:.6g}), sizes {r['sizes']}, offset {off:.10f}")
    for row in rows:
        N = row["N"]; e = row["r_extrap"] - off
        e = (e + math.pi / 2) % math.pi - math.pi / 2
        if ch["kind"] == "balanced":
            say(f"   N = {N:9.3f}   N*r = {N * e:+.8f}   predicted {ch['second_order']:+.8f}   size spread {abs(row['r'][2] - row['r'][1]):.1e}")
        elif ch["kind"] == "rough":
            say(f"   N = {N:9.3f}   N^delta*r = {N ** ch['delta'] * e:+.6f}   predicted {ch['sh_const']:+.6f}"
                f" (+ {(2 * al - 1) / 24 * J_alpha(al):.4f} N^(delta-1))   size spread {abs(row['r'][2] - row['r'][1]):.1e}")
        else:
            say(f"   N = {N:9.3f}   r = {e:+.3e}   N*r = {N * e:+.5f}   size spread {abs(row['r'][2] - row['r'][1]):.1e}")
open(os.path.join(OUT, "numerics_table.txt"), "w").write(chr(10).join(lines) + chr(10))
