#!/usr/bin/env python3
"""Welch t-tests (no scipy) on randomized-order N=5 telemetry."""
import csv, math, statistics as st, os, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "ablation_multitrial_confirmation.csv"))))

def betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-30 else 1e-30); h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + aa / c; c = c if abs(c) > 1e-30 else 1e-30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + aa / c; c = c if abs(c) > 1e-30 else 1e-30
        de = d * c; h *= de
        if abs(de - 1) < 3e-12: break
    return h

def ibeta(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2): return bt * betacf(a, b, x) / a
    return 1 - bt * betacf(b, a, 1 - x) / b

def welch(x, y):
    mx, my, vx, vy, nx, ny = st.mean(x), st.mean(y), st.variance(x), st.variance(y), len(x), len(y)
    se2 = vx / nx + vy / ny
    t = (mx - my) / math.sqrt(se2)
    df = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    p = ibeta(df / 2, 0.5, df / (df + t * t))
    return t, df, p

def get(prec, g, col="joules_per_1k"):
    return [float(r[col]) for r in rows if r["precision"] == prec and int(r["group_size"]) == g]

out = {}
pairs = [("fp16", 0, "fp32", 0), ("fp16", 0, "int4", 128), ("fp16", 0, "int4", 256), ("int4", 128, "fp32", 0),
         ("int4", 256, "int4", 128), ("int4", 128, "int4", 64), ("int4", 72, "int4", 128), ("int4", 72, "int4", 64)]
for a, ga, b, gb in pairs:
    t, df, p = welch(get(a, ga), get(b, gb))
    out[f"{a}{ga}_vs_{b}{gb}"] = dict(t=round(t, 2), df=round(df, 1), p=round(p, 4))
    print(f"{a}-G{ga} vs {b}-G{gb}: t={t:.2f} df={df:.1f} p={p:.4f}  means {st.mean(get(a,ga)):.4f} / {st.mean(get(b,gb)):.4f}")

# thermal confound check: correlation between temperature and energy within INT4
import itertools
xs = [float(r["gpu_temp_c"]) for r in rows if r["precision"] == "int4"]
ys = [float(r["joules_per_1k"]) for r in rows if r["precision"] == "int4"]
mx, my = st.mean(xs), st.mean(ys)
r = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
print(f"INT4 corr(temp, J/1k) = {r:.2f}")
ord_e = [(int(r["order_idx"]), float(r["joules_per_1k"])) for r in rows]
print("Idle baseline range (W):", min(float(r['idle_power_w']) for r in rows), max(float(r['idle_power_w']) for r in rows))
print("Throughput means:")
for prec, g in [("fp32", 0), ("fp16", 0), ("int8", 64)] + [("int4", x) for x in (32, 48, 64, 72, 96, 128, 256)]:
    v = get(prec, g, "dispatches_per_sec"); print(f"  {prec}-G{g}: {st.mean(v):.1f} ± {st.stdev(v):.1f}  E={st.mean(get(prec,g)):.4f} ± {st.stdev(get(prec,g)):.4f}")
json.dump(out, open(os.path.join(ROOT, "results", "energy_stats.json"), "w"), indent=2)
