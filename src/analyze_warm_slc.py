#!/usr/bin/env python3
"""Analyze replicated warm-SLC trials: power-of-two (PoT) vs non-PoT group sizes."""
import csv, math, random, statistics as st, sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/ablation_warm_slc_repeats.csv"
rows = list(csv.DictReader(open(PATH)))
POT = {16, 32, 64, 128, 256}

by = defaultdict(list)  # (kernel, G) -> gflops list
for r in rows:
    by[(r["kernel"], int(r["group_size"]))].append(float(r["gflops"]))

out = []
def ci(xs):
    m = st.mean(xs); s = st.stdev(xs); h = 2.09 * s / math.sqrt(len(xs))
    return m, s, h

lines = ["| Kernel | G | PoT | Mean GFLOPS | Std | 95% CI |", "| :--- | :--- | :--- | :--- | :--- | :--- |"]
for k in ("C1Tpl", "C4Tpl"):
    for g in sorted({g for (kk, g) in by if kk == k}):
        m, s, h = ci(by[(k, g)])
        lines.append(f"| {k} | {g} | {'yes' if g in POT else 'no'} | {m:.1f} | {s:.1f} | ±{h:.1f} |")

lines += ["", "| Kernel | PoT mean | non-PoT mean | PoT advantage | 95% bootstrap CI | p (perm) |", "| :--- | :--- | :--- | :--- | :--- | :--- |"]
random.seed(0)
for k in ("C1Tpl", "C4Tpl"):
    reps = sorted({int(r["rep"]) for r in rows})
    per_rep = []
    for rep in reps:
        pot = [float(r["gflops"]) for r in rows if r["kernel"] == k and int(r["rep"]) == rep and int(r["group_size"]) in POT]
        npt = [float(r["gflops"]) for r in rows if r["kernel"] == k and int(r["rep"]) == rep and int(r["group_size"]) not in POT]
        per_rep.append((st.mean(pot), st.mean(npt)))
    diffs = [(a - b) / b * 100 for a, b in per_rep]
    d = st.mean(diffs)
    boots = sorted(st.mean(random.choices(diffs, k=len(diffs))) for _ in range(5000))
    lo, hi = boots[125], boots[4874]
    # sign-flip permutation test on paired per-rep differences
    cnt = 0
    for _ in range(20000):
        if abs(st.mean([x if random.random() < .5 else -x for x in diffs])) >= abs(d):
            cnt += 1
    p = cnt / 20000
    lines.append(f"| {k} | {st.mean(a for a,_ in per_rep):.1f} | {st.mean(b for _,b in per_rep):.1f} | {d:+.2f}% | [{lo:+.2f}%, {hi:+.2f}%] | {p:.3f} |")

text = "\n".join(lines)
print(text)
open(PATH.replace(".csv", "_summary.md"), "w").write(text + "\n")
