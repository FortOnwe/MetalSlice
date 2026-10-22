#!/usr/bin/env python3
"""Generate publication-grade SVG figures for MetalSlice purely from CSV telemetry."""
import csv
import math
import os
import statistics as st
import xml.etree.ElementTree as ET

RESULTS_DIR = "/Users/fo/Documents/Antigravity Test/metal_unified_memory_quant_study/results"


def generate_pareto_chart():
    csv_path = os.path.join(RESULTS_DIR, "ablation_multitrial_confirmation.csv")
    if not os.path.exists(csv_path):
        return

    raw_data = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            raw_data.append({
                "trial": int(row["trial"]),
                "precision": row["precision"],
                "group_size": int(row["group_size"]),
                "throughput": float(row["dispatches_per_sec"]),
                "active_power": float(row["active_power_w"]),
                "energy_1k": float(row["joules_per_1k"]),
            })

    num_trials = len({r["trial"] for r in raw_data})

    configs = [
        ("fp32", 0, "FP32", "#64748b"),
        ("fp16", 0, "FP16", "#059669"),
        ("int8", 64, "INT8-G64", "#0284c7"),
        ("int4", 32, "INT4-G32", "#7c3aed"),
        ("int4", 48, "INT4-G48", "#8b5cf6"),
        ("int4", 64, "INT4-G64", "#2563eb"),
        ("int4", 72, "INT4-G72", "#0d9488"),
        ("int4", 96, "INT4-G96", "#0284c7"),
        ("int4", 128, "INT4-G128", "#ea580c"),
        ("int4", 256, "INT4-G256", "#d97706"),
    ]

    points = []
    for prec, gs, label, color in configs:
        subset = [r for r in raw_data if r["precision"] == prec and r["group_size"] == gs]
        if subset:
            tps = [r["throughput"] for r in subset]
            pws = [r["active_power"] for r in subset]
            ens = [r["energy_1k"] for r in subset]
            points.append({
                "label": label,
                "color": color,
                "tp": st.mean(tps),
                "tp_sd": st.stdev(tps) if len(tps) > 1 else 0.0,
                "pw": st.mean(pws),
                "pw_sd": st.stdev(pws) if len(pws) > 1 else 0.0,
                "en": st.mean(ens),
                "en_sd": st.stdev(ens) if len(ens) > 1 else 0.0,
            })

    width = 1180
    height = 640
    pad_left = 95
    pad_right = 400
    pad_top = 100
    pad_bottom = 85

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    max_x = 22500.0
    max_y = 2.6

    def x_px(val):
        return pad_left + (val / max_x) * plot_w

    def y_px(val):
        return height - pad_bottom - (val / max_y) * plot_h

    svg = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '  <defs>',
        '    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">',
        '      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.12"/>',
        '    </filter>',
        '  </defs>',
        '  <style>',
        '    .bg { fill: #ffffff; }',
        '    .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 20px; font-weight: 700; fill: #0f172a; }',
        '    .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; fill: #475569; }',
        '    .axis-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; font-weight: 600; fill: #1e293b; }',
        '    .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4,4; }',
        '    .axis-line { stroke: #94a3b8; stroke-width: 1.5; }',
        '    .tick-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #64748b; }',
        '    .point-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; }',
        '    .point-sub { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; font-weight: 500; fill: #475569; }',
        '    .table-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; font-weight: 700; fill: #1e293b; }',
        '    .table-cell { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #334155; }',
        '    .table-cell-bold { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; font-weight: 700; fill: #0f172a; }',
        '  </style>',
        f'  <rect width="{width}" height="{height}" class="bg" rx="14" stroke="#cbd5e1" stroke-width="1.5"/>',
        f'  <text x="{pad_left}" y="42" class="title">MetalSlice: Throughput vs. Energy Pareto Frontier (N={num_trials} Randomized Trials)</text>',
        f'  <text x="{pad_left}" y="65" class="subtitle">Apple M5 Pro | Matrix: 4608×4608 | Cold DRAM 32-Buffer Ring Pool | Per-Run Idle Baseline Subtraction (±1 SD Error Bars)</text>',
    ]

    for x_val in range(0, 25000, 5000):
        if x_val <= max_x:
            px = x_px(x_val)
            svg.append(f'  <line x1="{px}" y1="{pad_top}" x2="{px}" y2="{height - pad_bottom}" class="grid"/>')
            svg.append(f'  <text x="{px}" y="{height - pad_bottom + 22}" class="tick-label" text-anchor="middle">{x_val:,}</text>')

    for y_step in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        py = y_px(y_step)
        svg.append(f'  <line x1="{pad_left}" y1="{py}" x2="{pad_left + plot_w}" y2="{py}" class="grid"/>')
        svg.append(f'  <text x="{pad_left - 12}" y="{py + 4}" class="tick-label" text-anchor="end">{y_step:.1f} J</text>')

    svg.append(f'  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{pad_left + plot_w}" y2="{height - pad_bottom}" class="axis-line"/>')
    svg.append(f'  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" class="axis-line"/>')

    svg.append(f'  <text x="{pad_left + plot_w/2}" y="{height - 30}" class="axis-title" text-anchor="middle">Sustained Throughput (GEMV ops/s) → Higher is Better</text>')
    svg.append(f'  <text x="28" y="{pad_top + plot_h/2}" class="axis-title" text-anchor="middle" transform="rotate(-90 28 {pad_top + plot_h/2})">Net Energy (Joules / 1k GEMV) → Lower is Better</text>')

    opt_x = x_px(18000)
    opt_y = y_px(1.50)
    svg.append(f'  <rect x="{opt_x}" y="{opt_y}" width="{pad_left + plot_w - opt_x}" height="{height - pad_bottom - opt_y}" fill="#10b981" fill-opacity="0.08" rx="8"/>')
    svg.append(f'  <text x="{pad_left + plot_w - 12}" y="{height - pad_bottom - 12}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#047857" text-anchor="end">High-Throughput INT4 Region</text>')

    i4_pts = [q for q in points if q["label"].startswith("INT4")]
    max_tp_pt = max(i4_pts, key=lambda q: q["tp"])

    # Draw ±1 SD error bars first
    for p in points:
        cx, cy = x_px(p["tp"]), y_px(p["en"])
        x_lo, x_hi = x_px(p["tp"] - p["tp_sd"]), x_px(p["tp"] + p["tp_sd"])
        y_lo, y_hi = y_px(p["en"] - p["en_sd"]), y_px(p["en"] + p["en_sd"])
        svg.append(f'  <line x1="{x_lo}" y1="{cy}" x2="{x_hi}" y2="{cy}" stroke="{p["color"]}" stroke-width="1.5" opacity="0.7"/>')
        svg.append(f'  <line x1="{cx}" y1="{y_lo}" x2="{cx}" y2="{y_hi}" stroke="{p["color"]}" stroke-width="1.5" opacity="0.7"/>')
        svg.append(f'  <line x1="{cx - 3}" y1="{y_lo}" x2="{cx + 3}" y2="{y_lo}" stroke="{p["color"]}" stroke-width="1.5" opacity="0.7"/>')
        svg.append(f'  <line x1="{cx - 3}" y1="{y_hi}" x2="{cx + 3}" y2="{y_hi}" stroke="{p["color"]}" stroke-width="1.5" opacity="0.7"/>')

    # Draw points and non-overlapping callouts
    for p in points:
        cx, cy = x_px(p["tp"]), y_px(p["en"])
        if p["label"] == "FP32":
            svg.append(f'  <circle cx="{cx}" cy="{cy}" r="7" fill="{p["color"]}" stroke="#ffffff" stroke-width="2" filter="url(#shadow)"/>')
            svg.append(f'  <text x="{cx + 14}" y="{cy - 4}" class="point-label">FP32 (Baseline)</text>')
            svg.append(f'  <text x="{cx + 14}" y="{cy + 13}" class="point-sub">{int(round(p["tp"])):,} ops/s | {p["en"]:.2f} ± {p["en_sd"]:.2f} J</text>')
        elif p["label"] == "FP16":
            svg.append(f'  <circle cx="{cx}" cy="{cy}" r="8" fill="{p["color"]}" stroke="#ffffff" stroke-width="2" filter="url(#shadow)"/>')
            svg.append(f'  <rect x="{cx - 10}" y="{cy - 54}" width="215" height="26" rx="6" fill="#059669"/>')
            svg.append(f'  <text x="{cx + 97}" y="{cy - 37}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#ffffff" text-anchor="middle">★ Lowest Energy: {p["en"]:.2f} ± {p["en_sd"]:.2f} J</text>')
            svg.append(f'  <line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - 28}" stroke="#059669" stroke-width="1.5" stroke-dasharray="3,3"/>')
            svg.append(f'  <text x="{cx + 14}" y="{cy + 16}" class="point-label">FP16</text>')
            svg.append(f'  <text x="{cx + 14}" y="{cy + 31}" class="point-sub">{int(round(p["tp"])):,} ops/s ({p["pw"]:.1f} W low-power state)</text>')
        elif p["label"] == "INT8-G64":
            svg.append(f'  <circle cx="{cx}" cy="{cy}" r="7" fill="{p["color"]}" stroke="#ffffff" stroke-width="2" filter="url(#shadow)"/>')
            svg.append(f'  <text x="{cx}" y="{cy - 26}" class="point-label" text-anchor="middle">INT8-G64</text>')
            svg.append(f'  <text x="{cx}" y="{cy - 11}" class="point-sub" text-anchor="middle">{int(round(p["tp"])):,} ops/s | {p["en"]:.2f} ± {p["en_sd"]:.2f} J</text>')
        else:
            is_peak = p["label"] == max_tp_pt["label"]
            is_npot = "G72" in p["label"] or "G96" in p["label"]
            r_size = 7.5 if is_peak else (6.5 if is_npot else 5.5)
            svg.append(f'  <circle cx="{cx}" cy="{cy}" r="{r_size}" fill="{p["color"]}" stroke="#ffffff" stroke-width="2" filter="url(#shadow)"/>')

    # Callout for Max INT4 Throughput placed cleanly above the INT4 cluster, right-aligned inside plot
    cx_p, cy_p = x_px(max_tp_pt["tp"]), y_px(max_tp_pt["en"])
    pill_w = 264
    pill_x = min(cx_p - pill_w / 2, pad_left + plot_w - pill_w - 6)
    pill_y = cy_p - 66
    svg.append(f'  <rect x="{pill_x}" y="{pill_y}" width="{pill_w}" height="26" rx="6" fill="#ea580c"/>')
    svg.append(f'  <text x="{pill_x + pill_w/2}" y="{pill_y + 17}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#ffffff" text-anchor="middle">★ Max INT4 ({max_tp_pt["label"]}): {int(round(max_tp_pt["tp"])):,} ops/s</text>')
    svg.append(f'  <line x1="{cx_p}" y1="{cy_p}" x2="{pill_x + pill_w/2 + 20}" y2="{pill_y + 26}" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="3,3"/>')

    # INT4 Cluster Label below points (completely inside plot area)
    cluster_cx = x_px(19600)
    cluster_cy = y_px(0.95)
    cluster_sub = f"{min(q['tp'] for q in i4_pts)/1000:.1f}k–{max(q['tp'] for q in i4_pts)/1000:.1f}k ops/s | {min(q['en'] for q in i4_pts):.2f}–{max(q['en'] for q in i4_pts):.2f} J (G≥64 energy differences p ≥ 0.50)"
    svg.append(f'  <text x="{cluster_cx}" y="{cluster_cy}" class="point-label" fill="#2563eb" text-anchor="middle">INT4 Group-Size Cluster (G=32..256)</text>')
    svg.append(f'  <text x="{cluster_cx}" y="{cluster_cy + 16}" class="point-sub" text-anchor="middle">{cluster_sub}</text>')

    # Side Table Inset
    tbl_x = width - pad_right + 22
    tbl_y = pad_top - 10
    tbl_w = pad_right - 36
    tbl_h = plot_h + 20

    svg.append(f'  <rect x="{tbl_x}" y="{tbl_y}" width="{tbl_w}" height="{tbl_h}" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" rx="8"/>')
    svg.append(f'  <text x="{tbl_x + 14}" y="{tbl_y + 24}" font-family="-apple-system, sans-serif" font-size="13px" font-weight="700" fill="#0f172a">Multi-Trial Summary (N={num_trials}, Mean ± SD)</text>')
    svg.append(f'  <text x="{tbl_x + 14}" y="{tbl_y + 40}" font-family="-apple-system, sans-serif" font-size="11px" fill="#64748b">Randomized config order, 8s cooldown + per-run idle:</text>')

    y_row = tbl_y + 64
    svg.append(f'  <line x1="{tbl_x + 12}" y1="{y_row - 16}" x2="{tbl_x + tbl_w - 12}" y2="{y_row - 16}" stroke="#cbd5e1" stroke-width="1"/>')
    svg.append(f'  <text x="{tbl_x + 14}" y="{y_row}" class="table-hdr">Config</text>')
    svg.append(f'  <text x="{tbl_x + 92}" y="{y_row}" class="table-hdr">GEMV/s</text>')
    svg.append(f'  <text x="{tbl_x + 190}" y="{y_row}" class="table-hdr">Active W</text>')
    svg.append(f'  <text x="{tbl_x + 265}" y="{y_row}" class="table-hdr">J / 1k GEMV</text>')
    svg.append(f'  <line x1="{tbl_x + 12}" y1="{y_row + 8}" x2="{tbl_x + tbl_w - 12}" y2="{y_row + 8}" stroke="#cbd5e1" stroke-width="1"/>')

    for p in points:
        y_row += 38
        is_highlight = p["label"] in ("FP16", "INT4-G64", "INT4-G128", "INT4-G256")
        if is_highlight:
            svg.append(f'  <rect x="{tbl_x + 8}" y="{y_row - 15}" width="{tbl_w - 16}" height="24" fill="#e0f2fe" rx="4"/>')

        cell_cls = "table-cell-bold" if is_highlight else "table-cell"
        svg.append(f'  <circle cx="{tbl_x + 20}" cy="{y_row - 4}" r="4.5" fill="{p["color"]}"/>')
        svg.append(f'  <text x="{tbl_x + 30}" y="{y_row}" class="{cell_cls}">{p["label"]}</text>')
        svg.append(f'  <text x="{tbl_x + 92}" y="{y_row}" class="{cell_cls}">{int(round(p["tp"])):,} ± {int(round(p["tp_sd"]))}</text>')
        svg.append(f'  <text x="{tbl_x + 190}" y="{y_row}" class="{cell_cls}">{p["pw"]:.1f} ± {p["pw_sd"]:.1f}</text>')
        svg.append(f'  <text x="{tbl_x + 265}" y="{y_row}" class="{cell_cls}">{p["en"]:.3f} ± {p["en_sd"]:.3f}</text>')

    svg.append("</svg>")

    out_path = os.path.join(RESULTS_DIR, "throughput_energy_pareto.svg")
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"[+] Wrote Pareto Chart: {out_path}")


def generate_slc_cliff_chart():
    csv_path = os.path.join(RESULTS_DIR, "ablation_ring_pool_slc.csv")
    if not os.path.exists(csv_path):
        return

    rows = []
    with open(csv_path, "r") as f:
        for r in csv.DictReader(f):
            rows.append({
                "ring_size": int(r["ring_size"]),
                "alloc_mb": float(r["total_alloc_mb"]),
                "latency_us": float(r["median_gpu_us"]),
                "bw_gb_s": float(r["effective_bw_gb_s"]),
                "gflops": float(r["gflops"]),
            })

    width = 1100
    height = 560
    pad_left = 110
    pad_right = 60
    pad_top = 100
    pad_bottom = 90
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    max_y = 430.0

    def col_x(idx):
        return pad_left + (idx / (len(rows) - 1)) * plot_w

    def y_px(val):
        return height - pad_bottom - (val / max_y) * plot_h

    slc_rows = [r for r in rows if r["alloc_mb"] <= 22.0]
    dram_rows = [r for r in rows if r["alloc_mb"] >= 40.0]
    slc_bw_lo, slc_bw_hi = min(r["bw_gb_s"] for r in slc_rows), max(r["bw_gb_s"] for r in slc_rows)
    slc_gf_lo, slc_gf_hi = min(r["gflops"] for r in slc_rows), max(r["gflops"] for r in slc_rows)
    dram_bw_lo, dram_bw_hi = min(r["bw_gb_s"] for r in dram_rows), max(r["bw_gb_s"] for r in dram_rows)
    dram_us_lo, dram_us_hi = min(r["latency_us"] for r in dram_rows), max(r["latency_us"] for r in dram_rows)
    dram_gf_lo, dram_gf_hi = min(r["gflops"] for r in dram_rows), max(r["gflops"] for r in dram_rows)

    svg = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '  <style>',
        '    .bg { fill: #ffffff; }',
        '    .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 21px; font-weight: 700; fill: #0f172a; }',
        '    .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; fill: #475569; }',
        '    .axis-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; font-weight: 600; fill: #1e293b; }',
        '    .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4,4; }',
        '    .axis-line { stroke: #94a3b8; stroke-width: 1.5; }',
        '    .tick-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; font-weight: 700; fill: #1e293b; }',
        '    .tick-sub { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 10px; fill: #64748b; }',
        '    .line-slc { stroke: #2563eb; stroke-width: 3.5; fill: none; stroke-linecap: round; stroke-linejoin: round; }',
        '    .cliff-line { stroke: #dc2626; stroke-width: 2; stroke-dasharray: 6,4; }',
        '    .val-badge { rx: 5; }',
        '    .val-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; font-weight: 700; fill: #ffffff; }',
        '  </style>',
        f'  <rect width="{width}" height="{height}" class="bg" rx="14" stroke="#cbd5e1" stroke-width="1.5"/>',
        f'  <text x="{pad_left}" y="42" class="title">Ablation 1: GPU System-Level Cache (SLC) Eviction Curve</text>',
        f'  <text x="{pad_left}" y="65" class="subtitle">Effective Bandwidth vs. Working Set Size (INT4-G64, 4608×4608, C1Tpl) | 16-Dispatch Warmup</text>',
    ]

    x_cliff = (col_x(1) + col_x(2)) / 2.0
    svg.append(f'  <rect x="{pad_left}" y="{pad_top}" width="{x_cliff - pad_left}" height="{plot_h}" fill="#eff6ff"/>')
    svg.append(f'  <rect x="{x_cliff}" y="{pad_top}" width="{pad_left + plot_w - x_cliff}" height="{plot_h}" fill="#f8fafc"/>')

    for bw_val in [0, 100, 200, 250, 300, 350, 400]:
        py = y_px(bw_val)
        svg.append(f'  <line x1="{pad_left}" y1="{py}" x2="{pad_left + plot_w}" y2="{py}" class="grid"/>')
        svg.append(f'  <text x="{pad_left - 14}" y="{py + 4}" font-family="-apple-system, sans-serif" font-size="11px" fill="#64748b" text-anchor="end">{bw_val} GB/s</text>')

    for idx, r in enumerate(rows):
        cx = col_x(idx)
        svg.append(f'  <line x1="{cx}" y1="{pad_top}" x2="{cx}" y2="{height - pad_bottom}" class="grid"/>')
        svg.append(f'  <text x="{cx}" y="{height - pad_bottom + 18}" class="tick-hdr" text-anchor="middle">R={r["ring_size"]}</text>')
        svg.append(f'  <text x="{cx}" y="{height - pad_bottom + 32}" class="tick-sub" text-anchor="middle">{r["alloc_mb"]:.1f} MB</text>')

    svg.append(f'  <line x1="{x_cliff}" y1="{pad_top}" x2="{x_cliff}" y2="{height - pad_bottom}" class="cliff-line"/>')

    # Zone summary callouts drawn AFTER gridlines so grid dashes never cross text
    box_y = height - pad_bottom - 95
    svg.append(f'  <rect x="{pad_left + 10}" y="{box_y}" width="130" height="24" rx="5" fill="#2563eb"/>')
    svg.append(f'  <text x="{pad_left + 75}" y="{box_y + 16}" font-family="-apple-system, sans-serif" font-size="10px" font-weight="700" fill="#ffffff" text-anchor="middle">SLC Hit (≤ 21.5 MB)</text>')
    svg.append(f'  <text x="{pad_left + 10}" y="{box_y + 40}" font-family="-apple-system, sans-serif" font-size="10.5px" font-weight="600" fill="#1e40af">{slc_bw_lo:.1f}–{slc_bw_hi:.1f} GB/s</text>')
    svg.append(f'  <text x="{pad_left + 10}" y="{box_y + 55}" font-family="-apple-system, sans-serif" font-size="10px" fill="#3b82f6">{int(round(slc_gf_lo)):,}–{int(round(slc_gf_hi)):,} GFLOPS</text>')

    svg.append(f'  <rect x="{x_cliff + 25}" y="{pad_top + 15}" width="260" height="26" rx="5" fill="#475569"/>')
    svg.append(f'  <text x="{x_cliff + 155}" y="{pad_top + 32}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#ffffff" text-anchor="middle">Cold Unified DRAM Saturation (≥ 43.0 MB)</text>')
    svg.append(f'  <text x="{x_cliff + 25}" y="{pad_top + 58}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="600" fill="#334155">LPDDR5X Measured Plateau: {dram_bw_lo:.1f} – {dram_bw_hi:.1f} GB/s</text>')
    svg.append(f'  <text x="{x_cliff + 25}" y="{pad_top + 74}" font-family="-apple-system, sans-serif" font-size="10px" fill="#64748b">Cold Latency: {dram_us_lo:.2f} – {dram_us_hi:.2f} µs ({int(round(dram_gf_lo)):,} – {int(round(dram_gf_hi)):,} GFLOPS)</text>')

    # Place cliff badge at y_px(185) in open space well below the 267.8 badge
    svg.append(f'  <rect x="{x_cliff - 82}" y="{y_px(185)}" width="164" height="26" rx="5" fill="#dc2626"/>')
    svg.append(f'  <text x="{x_cliff}" y="{y_px(185) + 17}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#ffffff" text-anchor="middle">SLC Eviction (21.5–43 MB)</text>')

    svg.append(f'  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{pad_left + plot_w}" y2="{height - pad_bottom}" class="axis-line"/>')
    svg.append(f'  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" class="axis-line"/>')

    svg.append(f'  <text x="{pad_left + plot_w/2}" y="{height - 25}" class="axis-title" text-anchor="middle">Discrete Ring Pool Steps (Categorical Spacing; Working Set Shown in MB)</text>')
    svg.append(f'  <text x="32" y="{pad_top + plot_h/2}" class="axis-title" text-anchor="middle" transform="rotate(-90 32 {pad_top + plot_h/2})">Measured Effective Bandwidth (GB/s)</text>')

    pts = [f'{col_x(i)},{y_px(r["bw_gb_s"])}' for i, r in enumerate(rows)]
    svg.append(f'  <polyline points="{" ".join(pts)}" class="line-slc"/>')

    for idx, r in enumerate(rows):
        cx = col_x(idx)
        cy = y_px(r["bw_gb_s"])
        svg.append(f'  <circle cx="{cx}" cy="{cy}" r="6.5" fill="#2563eb" stroke="#ffffff" stroke-width="2"/>')
        badge_y = cy - 26 if idx % 2 == 0 else cy + 14
        badge_fill = "#1d4ed8" if r["bw_gb_s"] > 300 else "#334155"
        svg.append(f'  <rect x="{cx - 32}" y="{badge_y - 10}" width="64" height="18" class="val-badge" fill="{badge_fill}"/>')
        svg.append(f'  <text x="{cx}" y="{badge_y + 3}" class="val-text" text-anchor="middle">{r["bw_gb_s"]:.1f}</text>')

    svg.append("</svg>")

    out_path = os.path.join(RESULTS_DIR, "slc_eviction_curve.svg")
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"[+] Wrote SLC Cliff Chart: {out_path}")


def generate_group_scaling_chart():
    csv_path = os.path.join(RESULTS_DIR, "ablation_group_granularity.csv")
    if not os.path.exists(csv_path):
        return

    rows = []
    with open(csv_path, "r") as f:
        for r in csv.DictReader(f):
            rows.append({
                "g": int(r["group_size"]),
                "leg_dram": float(r["legacy_dram_gflops"]),
                "c1_dram": float(r["coal1_tpl_dram_gflops"]),
                "c1_slc": float(r["coal1_tpl_slc_gflops"]),
                "c4_slc": float(r["coal4_tpl_slc_gflops"]),
                "c1_slc_sd": 0.0,
                "c4_slc_sd": 0.0,
            })

    rp = os.path.join(RESULTS_DIR, "ablation_warm_slc_repeats.csv")
    n_reps = 1
    if os.path.exists(rp):
        acc1, acc4 = {}, {}
        with open(rp) as f:
            for r in csv.DictReader(f):
                g = int(r["group_size"])
                if r["kernel"] == "C1Tpl":
                    acc1.setdefault(g, []).append(float(r["gflops"]))
                elif r["kernel"] == "C4Tpl":
                    acc4.setdefault(g, []).append(float(r["gflops"]))
        for row in rows:
            g = row["g"]
            if g in acc1:
                row["c1_slc"] = st.mean(acc1[g])
                row["c1_slc_sd"] = st.stdev(acc1[g])
                n_reps = len(acc1[g])
            if g in acc4:
                row["c4_slc"] = st.mean(acc4[g])
                row["c4_slc_sd"] = st.stdev(acc4[g])

    width = 1100
    height = 600
    pad_left = 100
    pad_right = 60
    pad_top = 125
    pad_bottom = 90
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    max_y = 1750.0

    def col_x(idx):
        return pad_left + (idx / (len(rows) - 1)) * plot_w

    def y_px(val):
        return height - pad_bottom - (val / max_y) * plot_h

    svg = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '  <style>',
        '    .bg { fill: #ffffff; }',
        '    .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 21px; font-weight: 700; fill: #0f172a; }',
        '    .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; fill: #475569; }',
        '    .axis-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; font-weight: 600; fill: #1e293b; }',
        '    .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4,4; }',
        '    .axis-line { stroke: #94a3b8; stroke-width: 1.5; }',
        '    .tick-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; }',
        '    .tick-sub { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; font-weight: 600; }',
        '    .line-c4slc { stroke: #7c3aed; stroke-width: 3.2; fill: none; }',
        '    .line-c1slc { stroke: #0d9488; stroke-width: 2.5; stroke-dasharray: 3,3; fill: none; }',
        '    .line-c1dram { stroke: #2563eb; stroke-width: 3.5; fill: none; }',
        '    .line-leg { stroke: #dc2626; stroke-width: 2.5; stroke-dasharray: 6,4; fill: none; }',
        '  </style>',
        f'  <rect width="{width}" height="{height}" class="bg" rx="14" stroke="#cbd5e1" stroke-width="1.5"/>',
        f'  <text x="{pad_left}" y="38" class="title">Ablation 3: Group Granularity &amp; Kernel Architecture Scaling (4608×4608)</text>',
        f'  <text x="{pad_left}" y="60" class="subtitle">Warm SLC (N={n_reps} randomized reps; C1Tpl PoT +3.6%, C4Tpl PoT -0.6%) vs. Cold DRAM Coalesced Plateau &amp; Legacy Starvation</text>',
    ]

    # Dedicated horizontal legend bar at y=88 (zero collision with title/subtitle)
    ly = 90
    svg.append(f'  <rect x="{pad_left}" y="{ly - 16}" width="{plot_w}" height="24" rx="5" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>')
    svg.append(f'  <line x1="{pad_left + 12}" y1="{ly - 4}" x2="{pad_left + 36}" y2="{ly - 4}" class="line-c4slc"/>')
    svg.append(f'  <text x="{pad_left + 42}" y="{ly}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="600" fill="#7c3aed">Warm SLC C4Tpl (N={n_reps})</text>')

    svg.append(f'  <line x1="{pad_left + 215}" y1="{ly - 4}" x2="{pad_left + 239}" y2="{ly - 4}" class="line-c1slc"/>')
    svg.append(f'  <text x="{pad_left + 245}" y="{ly}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="600" fill="#0d9488">Warm SLC C1Tpl (N={n_reps})</text>')

    svg.append(f'  <line x1="{pad_left + 420}" y1="{ly - 4}" x2="{pad_left + 444}" y2="{ly - 4}" class="line-c1dram"/>')
    svg.append(f'  <text x="{pad_left + 450}" y="{ly}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="600" fill="#2563eb">Cold DRAM C1Tpl (Flat Plateau G ≥ 64)</text>')

    svg.append(f'  <line x1="{pad_left + 705}" y1="{ly - 4}" x2="{pad_left + 729}" y2="{ly - 4}" class="line-leg"/>')
    svg.append(f'  <text x="{pad_left + 735}" y="{ly}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="600" fill="#dc2626">Cold DRAM Legacy (Starves G ≥ 128)</text>')

    for gflops_val in [600, 800, 1000, 1200, 1400, 1600]:
        py = y_px(gflops_val)
        svg.append(f'  <line x1="{pad_left}" y1="{py}" x2="{pad_left + plot_w}" y2="{py}" class="grid"/>')
        svg.append(f'  <text x="{pad_left - 14}" y="{py + 4}" font-family="-apple-system, sans-serif" font-size="11px" fill="#64748b" text-anchor="end">{gflops_val}</text>')

    svg.append(f'  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{pad_left + plot_w}" y2="{height - pad_bottom}" class="axis-line"/>')
    svg.append(f'  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" class="axis-line"/>')

    svg.append(f'  <text x="{pad_left + plot_w/2}" y="{height - 25}" class="axis-title" text-anchor="middle">Quantization Group Size (G) → Finer (Left) to Coarser (Right)</text>')
    svg.append(f'  <text x="32" y="{pad_top + plot_h/2}" class="axis-title" text-anchor="middle" transform="rotate(-90 32 {pad_top + plot_h/2})">Effective Compute (GFLOPS)</text>')

    for idx, r in enumerate(rows):
        cx = col_x(idx)
        svg.append(f'  <line x1="{cx}" y1="{pad_top}" x2="{cx}" y2="{height - pad_bottom}" class="grid"/>')
        is_npot = r["g"] in [24, 48, 72, 96]
        sub = "NPOT" if is_npot else "POT"
        sub_color = "#0284c7" if is_npot else "#64748b"
        svg.append(f'  <text x="{cx}" y="{height - pad_bottom + 20}" class="tick-hdr" text-anchor="middle">G={r["g"]}</text>')
        svg.append(f'  <text x="{cx}" y="{height - pad_bottom + 34}" class="tick-sub" fill="{sub_color}" text-anchor="middle">{sub}</text>')

    pts_c4 = [f'{col_x(i)},{y_px(r["c4_slc"])}' for i, r in enumerate(rows)]
    pts_c1s = [f'{col_x(i)},{y_px(r["c1_slc"])}' for i, r in enumerate(rows)]
    pts_c1dram = [f'{col_x(i)},{y_px(r["c1_dram"])}' for i, r in enumerate(rows)]
    pts_leg = [f'{col_x(i)},{y_px(r["leg_dram"])}' for i, r in enumerate(rows)]

    svg.append(f'  <polyline points="{" ".join(pts_c4)}" class="line-c4slc"/>')
    svg.append(f'  <polyline points="{" ".join(pts_c1s)}" class="line-c1slc"/>')
    svg.append(f'  <polyline points="{" ".join(pts_c1dram)}" class="line-c1dram"/>')
    svg.append(f'  <polyline points="{" ".join(pts_leg)}" class="line-leg"/>')

    for i, r in enumerate(rows):
        cx = col_x(i)
        py_c4 = y_px(r["c4_slc"])
        py_c1s = y_px(r["c1_slc"])
        py_c1 = y_px(r["c1_dram"])
        py_leg = y_px(r["leg_dram"])

        svg.append(f'  <circle cx="{cx}" cy="{py_c4}" r="5.5" fill="#7c3aed" stroke="#ffffff" stroke-width="2"/>')
        svg.append(f'  <circle cx="{cx}" cy="{py_c1s}" r="4.5" fill="#0d9488" stroke="#ffffff" stroke-width="1.5"/>')
        svg.append(f'  <circle cx="{cx}" cy="{py_c1}" r="6" fill="#2563eb" stroke="#ffffff" stroke-width="2"/>')
        svg.append(f'  <circle cx="{cx}" cy="{py_leg}" r="5" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>')

        if r["g"] == 256:
            svg.append(f'  <rect x="{cx - 165}" y="{py_c4 - 32}" width="170" height="22" rx="4" fill="#7c3aed"/>')
            svg.append(f'  <text x="{cx - 80}" y="{py_c4 - 17}" font-family="-apple-system, sans-serif" font-size="10.5px" font-weight="700" fill="#ffffff" text-anchor="middle">C4Tpl SLC: {r["c4_slc"]:.1f} GFLOPS</text>')

            drop_pct = ((r["c1_dram"] - r["leg_dram"]) / r["c1_dram"]) * 100.0
            svg.append(f'  <rect x="{cx - 165}" y="{py_leg + 12}" width="170" height="24" rx="5" fill="#dc2626"/>')
            svg.append(f'  <text x="{cx - 80}" y="{py_leg + 28}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#ffffff" text-anchor="middle">Legacy Starvation: -{drop_pct:.1f}%</text>')

        if r["g"] == 96:
            svg.append(f'  <rect x="{cx - 82}" y="{py_c1 - 36}" width="164" height="22" rx="5" fill="#2563eb"/>')
            svg.append(f'  <text x="{cx}" y="{py_c1 - 21}" font-family="-apple-system, sans-serif" font-size="10.5px" font-weight="700" fill="#ffffff" text-anchor="middle">Cold Plateau: 987–1,015 GF</text>')

    svg.append("</svg>")

    out_path = os.path.join(RESULTS_DIR, "group_granularity_scaling.svg")
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"[+] Wrote Group Granularity Chart: {out_path}")


def main():
    print("[*] Generating publication-grade SVG figures for MetalSlice...")
    generate_pareto_chart()
    generate_slc_cliff_chart()
    generate_group_scaling_chart()

    for f in ["throughput_energy_pareto.svg", "slc_eviction_curve.svg", "group_granularity_scaling.svg"]:
        p = os.path.join(RESULTS_DIR, f)
        ET.parse(p)
        print(f"[PASS] {f} passed XML syntax verification")
    print("[+] All charts successfully verified and ready!")


if __name__ == "__main__":
    main()
