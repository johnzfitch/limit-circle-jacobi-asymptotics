"""Writes outputs/numerics_table.tex (rows of Table 2 of the paper) from outputs/trunc_cases/*.json.
Columns: chain, t, quantity, value at the largest level, prediction, largest N."""
import json, glob, math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from chains import CHAINS, J_alpha
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
order = ["B1", "B2", "B3", "SQ5m0", "SQ5m3", "SQ6m1", "RS", "D1", "D2", "BD6"]
tname = {0.0: "$0$", 1.0: "$1$", round(math.pi / 2, 6): r"$\pi/2$"}
rows = []
cases = {}
for f in glob.glob(os.path.join(OUT, "trunc_cases", "*.json")):
    r = json.load(open(f)); cases[(r["chain"], round(r["t"], 6))] = r
for name in order:
    for t in (0.0, 1.0, round(math.pi / 2, 6)):
        if (name, t) not in cases:
            continue
        r = cases[(name, t)]; ch = CHAINS[name]; al = ch["alpha"]; kp = ch.get("kappap", 0.0)
        if ch["kind"] == "dimer":
            th0 = kp * math.pi / (2 * (al - 1))
            off = th0 if t == 0.0 else (math.pi / 2 - th0 if (abs(t - math.pi / 2) < 1e-5 or kp > 0) else th0)
        else:
            off = r["t"]
        last = r["rows"][-1]; N = last["N"]
        e = (last["r_extrap"] - off + math.pi / 2) % math.pi - math.pi / 2
        if ch["kind"] == "balanced":
            q, val, pred = "$N r_j$", f"${N * e:.8f}$", f"${ch['second_order']:.8f}$"
        elif ch["kind"] == "rough":
            K = (2 * al - 1) / 24 * J_alpha(al)
            q, val, pred = "$N^{1/2} r_j$", f"{N ** ch['delta'] * e:.5f}", f"{ch['sh_const'] + K * N ** (ch['delta'] - 1):.5f}"
        else:
            q, val, pred = "$r_j$", f"{e:.2e}".replace("e-0", r"\cdot10^{-").replace("e+0", r"\cdot10^{") + "}", "$0$"
            val = "$" + val + "$"
        rows.append(f"{name} & {tname.get(round(r['t'], 6), r['t'])} & {q} & {val} & {pred} & {N:.0f}\\\\")
BS = chr(92)
head = [BS + "begin{tabular}{llllll}", BS + "hline", "chain & $t$ & quantity & measured & predicted & $N_" + BS + "lambda$" + BS + BS, BS + "hline"]
tail = [BS + "hline", BS + "end{tabular}"]
open(os.path.join(OUT, "numerics_table.tex"), "w").write(chr(10).join(head + rows + tail) + chr(10))
print(chr(10).join(rows))
