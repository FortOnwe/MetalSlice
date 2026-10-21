#!/usr/bin/env python3
import csv
import json
import os
import xml.etree.ElementTree as ET

RESULTS_DIR = "/Users/fo/Documents/Antigravity Test/metal_unified_memory_quant_study/results"

def generate_pareto_chart():
    csv_path = os.path.join(RESULTS_DIR, "ablation_multitrial_confirmation.csv")
    if not os.path.exists(csv_path):
        return

    raw_data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_data.append({
                "precision": row["precision"],
                "group_size": int(row["group_size"]),
                "throughput": float(row["dispatches_per_sec"]),
                "active_power": float(row["active_power_w"]),
                "energy_1k": float(row["joules_per_1k"])
            })

    configs = [
        ("fp32", 0, "FP32", "#64748b", "Baseline Reference"),
        ("fp16", 0, "FP16", "#059669", "Global Energy Min"),
        ("int8", 64, "INT8-G64", "#0284c7", "DVFS Boost Tax"),
        ("int4", 32, "INT4-G32", "#7c3aed", "12.5% scale overhead"),
        ("int4", 48, "INT4-G48", "#8b5cf6", "8.3% scale overhead"),
        ("int4", 64, "INT4-G64", "#2563eb", "Quantized energy min"),
        ("int4", 72, "INT4-G72", "#0d9488", "NPOT Sweet Spot"),
        ("int4", 96, "INT4-G96", "#0284c7", "NPOT Sweet Spot"),
        ("int4", 128, "INT4-G128", "#ea580c", "Peak Throughput"),
        ("int4", 256, "INT4-G256", "#d97706", "1.56% scale overhead")
    ]

    points = []
    for prec, gs, label, color, note in configs:
        subset = [r for r in raw_data if r["precision"] == prec and r["group_size"] == gs]
        if subset:
            mean_tp = sum(r["throughput"] for r in subset) / len(subset)
            mean_pw = sum(r["active_power"] for r in subset) / len(subset)
            mean_en = sum(r["energy_1k"] for r in subset) / len(subset)
            points.append({
                "label": label,
                "color": color,
                "note": note,
                "tp": mean_tp,
                "pw": mean_pw,
                "en": mean_en
            })

    width = 1150
    height = 640
    pad_left = 100
    pad_right = 360  # generous right margin for clean, spacious INT4 data table inset
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
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        '  <defs>',
        '    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">',
        '      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.12"/>',
        '    </filter>',
        '  </defs>',
        '  <style>',
        '    .bg { fill: #ffffff; }',
        '    .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 21px; font-weight: 700; fill: #0f172a; }',
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
        f'  <text x="{pad_left}" y="42" class="title">MetalSlice: Throughput vs. Energy Pareto Frontier</text>',
        f'  <text x="{pad_left}" y="65" class="subtitle">Apple Silicon Unified Memory | Matrix: 4608×4608 | Cold DRAM 32-Buffer Ring Pool | Hardware Timestamps</text>'
    ]

    # Gridlines X
    for x_val in range(0, 25000, 5000):
        if x_val <= max_x:
            px = x_px(x_val)
            svg.append(f'  <line x1="{px}" y1="{pad_top}" x2="{px}" y2="{height - pad_bottom}" class="grid"/>')
            svg.append(f'  <text x="{px}" y="{height - pad_bottom + 22}" class="tick-label" text-anchor="middle">{x_val:,}</text>')

    # Gridlines Y
    for y_step in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        py = y_px(y_step)
        svg.append(f'  <line x1="{pad_left}" y1="{py}" x2="{pad_left + plot_w}" y2="{py}" class="grid"/>')
        svg.append(f'  <text x="{pad_left - 12}" y="{py + 4}" class="tick-label" text-anchor="end">{y_step:.1f} J</text>')

    # Axis Lines
    svg.append(f'  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{pad_left + plot_w}" y2="{height - pad_bottom}" class="axis-line"/>')
    svg.append(f'  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" class="axis-line"/>')

    # Axis Titles
    svg.append(f'  <text x="{pad_left + plot_w/2}" y="{height - 30}" class="axis-title" text-anchor="middle">Sustained Throughput (GEMV ops/s) → Higher is Better</text>')
    svg.append(f'  <text x="30" y="{pad_top + plot_h/2}" class="axis-title" text-anchor="middle" transform="rotate(-90 30 {pad_top + plot_h/2})">Net Energy (Joules / 1k GEMV) → Lower is Better</text>')

    # Optimal Zone Highlight
    opt_x = x_px(17500)
    opt_y = y_px(1.4)
    svg.append(f'  <rect x="{opt_x}" y="{opt_y}" width="{pad_left + plot_w - opt_x}" height="{height - pad_bottom - opt_y}" fill="#10b981" fill-opacity="0.08" rx="8"/>')
    svg.append(f'  <text x="{pad_left + plot_w - 12}" y="{height - pad_bottom - 12}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#047857" text-anchor="end">Optimal Sweet Spot: High Speed, Low Energy</text>')

    # Non-INT4 Major Points Callouts (FP32, FP16, INT8)
    for p in points:
        cx = x_px(p["tp"])
        cy = y_px(p["en"])

        if p["label"] == "FP32":
            svg.append(f'  <circle cx="{cx}" cy="{cy}" r="7" fill="{p["color"]}" stroke="#ffffff" stroke-width="2" filter="url(#shadow)"/>')
            svg.append(f'  <text x="{cx + 14}" y="{cy - 4}" class="point-label">FP32 (Baseline)</text>')
            svg.append(f'  <text x="{cx + 14}" y="{cy + 13}" class="point-sub">{int(round(p["tp"])):,} ops/s | {p["en"]:.2f} J</text>')
        elif p["label"] == "FP16":
            svg.append(f'  <circle cx="{cx}" cy="{cy}" r="8" fill="{p["color"]}" stroke="#ffffff" stroke-width="2" filter="url(#shadow)"/>')
            # Callout Pill
            svg.append(f'  <rect x="{cx - 10}" y="{cy - 52}" width="200" height="26" rx="6" fill="#059669"/>')
            svg.append(f'  <text x="{cx + 90}" y="{cy - 35}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#ffffff" text-anchor="middle">★ Global Energy Min: 0.897 J</text>')
            svg.append(f'  <line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - 26}" stroke="#059669" stroke-width="1.5" stroke-dasharray="3,3"/>')
            svg.append(f'  <text x="{cx + 14}" y="{cy + 16}" class="point-label">FP16</text>')
            svg.append(f'  <text x="{cx + 14}" y="{cy + 30}" class="point-sub">{int(round(p["tp"])):,} ops/s (low-voltage state)</text>')
        elif p["label"] == "INT8-G64":
            svg.append(f'  <circle cx="{cx}" cy="{cy}" r="7" fill="{p["color"]}" stroke="#ffffff" stroke-width="2" filter="url(#shadow)"/>')
            svg.append(f'  <text x="{cx + 14}" y="{cy - 4}" class="point-label">INT8-G64</text>')
            svg.append(f'  <text x="{cx + 14}" y="{cy + 13}" class="point-sub">{int(round(p["tp"])):,} ops/s | {p["en"]:.2f} J (DVFS boost penalty)</text>')
        else:
            # INT4 points: draw crisp circles on the scatter plot
            is_peak = "G128" in p["label"]
            is_npot = "G72" in p["label"] or "G96" in p["label"]
            r_size = 8 if is_peak else (7 if is_npot else 5.5)
            svg.append(f'  <circle cx="{cx}" cy="{cy}" r="{r_size}" fill="{p["color"]}" stroke="#ffffff" stroke-width="2" filter="url(#shadow)"/>')

    # Highlighted Callout for INT4 Peak
    int4_128 = next(p for p in points if "G128" in p["label"])
    cx_128, cy_128 = x_px(int4_128["tp"]), y_px(int4_128["en"])
    svg.append(f'  <rect x="{cx_128 - 230}" y="{cy_128 - 48}" width="220" height="26" rx="6" fill="#ea580c"/>')
    svg.append(f'  <text x="{cx_128 - 120}" y="{cy_128 - 31}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#ffffff" text-anchor="middle">★ Throughput Peak: 20,418 ops/s</text>')
    svg.append(f'  <line x1="{cx_128}" y1="{cy_128}" x2="{cx_128 - 10}" y2="{cy_128 - 22}" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="3,3"/>')

    # INT4 Cluster Label on Plot
    svg.append(f'  <text x="{x_px(19500)}" y="{y_px(1.08)}" class="point-label" fill="#2563eb" text-anchor="middle">INT4 Group-Size Frontier</text>')
    svg.append(f'  <text x="{x_px(19500)}" y="{y_px(1.08) + 16}" class="point-sub" text-anchor="middle">18.9k – 20.4k ops/s | 1.21 – 1.27 J</text>')

    # -------------------------------------------------------------
    # Spacious Side Table Inset (Resolves INT4 Crowding Completely)
    # -------------------------------------------------------------
    tbl_x = width - pad_right + 30
    tbl_y = pad_top - 10
    tbl_w = pad_right - 45
    tbl_h = plot_h + 20

    svg.append(f'  <rect x="{tbl_x}" y="{tbl_y}" width="{tbl_w}" height="{tbl_h}" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" rx="8"/>')
    svg.append(f'  <text x="{tbl_x + 14}" y="{tbl_y + 24}" font-family="-apple-system, sans-serif" font-size="13px" font-weight="700" fill="#0f172a">INT4 Group Granularity Details</text>')
    svg.append(f'  <text x="{tbl_x + 14}" y="{tbl_y + 40}" font-family="-apple-system, sans-serif" font-size="11px" fill="#64748b">Measured multi-trial averages (N=3):</text>')

    # Table Header
    y_row = tbl_y + 64
    svg.append(f'  <line x1="{tbl_x + 12}" y1="{y_row - 16}" x2="{tbl_x + tbl_w - 12}" y2="{y_row - 16}" stroke="#cbd5e1" stroke-width="1"/>')
    svg.append(f'  <text x="{tbl_x + 16}" y="{y_row}" class="table-hdr">Config</text>')
    svg.append(f'  <text x="{tbl_x + 95}" y="{y_row}" class="table-hdr">Throughput</text>')
    svg.append(f'  <text x="{tbl_x + 185}" y="{y_row}" class="table-hdr">Active Pwr</text>')
    svg.append(f'  <text x="{tbl_x + 250}" y="{y_row}" class="table-hdr">Joules / 1k</text>')
    svg.append(f'  <line x1="{tbl_x + 12}" y1="{y_row + 8}" x2="{tbl_x + tbl_w - 12}" y2="{y_row + 8}" stroke="#cbd5e1" stroke-width="1"/>')

    int4_pts = [p for p in points if "INT4" in p["label"]]
    for idx, p in enumerate(int4_pts):
        y_row += 30
        is_highlight = "G128" in p["label"] or "G72" in p["label"]
        if is_highlight:
            svg.append(f'  <rect x="{tbl_x + 10}" y="{y_row - 15}" width="{tbl_w - 20}" height="24" fill="#e0f2fe" rx="4"/>')

        cell_cls = "table-cell-bold" if is_highlight else "table-cell"
        svg.append(f'  <circle cx="{tbl_x + 22}" cy="{y_row - 4}" r="4.5" fill="{p["color"]}"/>')
        svg.append(f'  <text x="{tbl_x + 32}" y="{y_row}" class="{cell_cls}">{p["label"].replace("INT4-", "")}</text>')
        svg.append(f'  <text x="{tbl_x + 95}" y="{y_row}" class="{cell_cls}">{int(round(p["tp"])):,} ops/s</text>')
        svg.append(f'  <text x="{tbl_x + 185}" y="{y_row}" class="{cell_cls}">{p["pw"]:.1f} W</text>')
        svg.append(f'  <text x="{tbl_x + 250}" y="{y_row}" class="{cell_cls}">{p["en"]:.4f} J</text>')

    svg.append('</svg>')

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
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "ring_size": int(r["ring_size"]),
                "alloc_mb": float(r["total_alloc_mb"]),
                "latency_us": float(r["median_gpu_us"]),
                "bw_gb_s": float(r["effective_bw_gb_s"]),
                "gflops": float(r["gflops"])
            })

    # Total 11 data points: use discrete evenly-spaced columns so points never crowd!
    width = 1100
    height = 560
    pad_left = 110
    pad_right = 60
    pad_top = 100
    pad_bottom = 90
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    max_y = 420.0

    def col_x(idx):
        return pad_left + (idx / (len(rows) - 1)) * plot_w

    def y_px(val):
        return height - pad_bottom - (val / max_y) * plot_h

    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
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
        f'  <text x="{pad_left}" y="65" class="subtitle">Effective Bandwidth vs. Working Set Size (INT4-G64, 4608×4608) | 16-Dispatch Warmup</text>'
    ]

    # Shaded Zones:
    # Zone 1: SLC Hit Zone (from col 0 to col 1.5)
    x_cliff = (col_x(1) + col_x(2)) / 2.0
    svg.append(f'  <rect x="{pad_left}" y="{pad_top}" width="{x_cliff - pad_left}" height="{plot_h}" fill="#eff6ff"/>')
    svg.append(f'  <rect x="{pad_left + 15}" y="{pad_top + 15}" width="190" height="26" rx="5" fill="#2563eb"/>')
    svg.append(f'  <text x="{pad_left + 110}" y="{pad_top + 32}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#ffffff" text-anchor="middle">On-Die SLC Hit Zone (≤ 21.5 MB)</text>')
    svg.append(f'  <text x="{pad_left + 15}" y="{pad_top + 58}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="600" fill="#1e40af">Bandwidth: 338.6 – 370.7 GB/s</text>')
    svg.append(f'  <text x="{pad_left + 15}" y="{pad_top + 74}" font-family="-apple-system, sans-serif" font-size="10px" fill="#3b82f6">Compute: 1,273 – 1,393 GFLOPS</text>')

    # Zone 2: Cold DRAM Saturation Zone (from x_cliff to end)
    svg.append(f'  <rect x="{x_cliff}" y="{pad_top}" width="{pad_left + plot_w - x_cliff}" height="{plot_h}" fill="#f8fafc"/>')
    svg.append(f'  <rect x="{x_cliff + 25}" y="{pad_top + 15}" width="250" height="26" rx="5" fill="#475569"/>')
    svg.append(f'  <text x="{x_cliff + 150}" y="{pad_top + 32}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#ffffff" text-anchor="middle">Cold Unified DRAM Saturation Zone (≥ 43.0 MB)</text>')
    svg.append(f'  <text x="{x_cliff + 25}" y="{pad_top + 58}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="600" fill="#334155">LPDDR5X Physical Bus Ceiling: ~234 – 251 GB/s</text>')
    svg.append(f'  <text x="{x_cliff + 25}" y="{pad_top + 74}" font-family="-apple-system, sans-serif" font-size="10px" fill="#64748b">Cold Latency: 44.9 – 48.3 µs (880 – 945 GFLOPS)</text>')

    # Eviction Cliff Marker
    svg.append(f'  <line x1="{x_cliff}" y1="{pad_top}" x2="{x_cliff}" y2="{height - pad_bottom}" class="cliff-line"/>')
    svg.append(f'  <rect x="{x_cliff - 75}" y="{y_px(300)}" width="150" height="26" rx="5" fill="#dc2626"/>')
    svg.append(f'  <text x="{x_cliff}" y="{y_px(300) + 17}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#ffffff" text-anchor="middle">SLC Eviction Cliff (~35 MB)</text>')

    # Gridlines Y
    for bw_val in [0, 100, 200, 250, 300, 350, 400]:
        py = y_px(bw_val)
        svg.append(f'  <line x1="{pad_left}" y1="{py}" x2="{pad_left + plot_w}" y2="{py}" class="grid"/>')
        svg.append(f'  <text x="{pad_left - 14}" y="{py + 4}" font-family="-apple-system, sans-serif" font-size="11px" fill="#64748b" text-anchor="end">{bw_val} GB/s</text>')

    # Axis Lines
    svg.append(f'  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{pad_left + plot_w}" y2="{height - pad_bottom}" class="axis-line"/>')
    svg.append(f'  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" class="axis-line"/>')

    # Axis Titles
    svg.append(f'  <text x="{pad_left + plot_w/2}" y="{height - 25}" class="axis-title" text-anchor="middle">Discrete Ring Pool Steps (Log-Quasi-Log Working Set Scale, MB) → Larger Defeats On-Die Cache</text>')
    svg.append(f'  <text x="32" y="{pad_top + plot_h/2}" class="axis-title" text-anchor="middle" transform="rotate(-90 32 {pad_top + plot_h/2})">Measured Effective Bandwidth (GB/s)</text>')

    # X-Axis Ticks (Columns)
    for idx, r in enumerate(rows):
        cx = col_x(idx)
        svg.append(f'  <line x1="{cx}" y1="{pad_top}" x2="{cx}" y2="{height - pad_bottom}" class="grid"/>')
        svg.append(f'  <text x="{cx}" y="{height - pad_bottom + 18}" class="tick-hdr" text-anchor="middle">R={r["ring_size"]}</text>')
        svg.append(f'  <text x="{cx}" y="{height - pad_bottom + 32}" class="tick-sub" text-anchor="middle">{r["alloc_mb"]:.1f} MB</text>')

    # Polyline Path
    pts = [f'{col_x(i)},{y_px(r["bw_gb_s"])}' for i, r in enumerate(rows)]
    svg.append(f'  <polyline points="{" ".join(pts)}" class="line-slc"/>')

    # Plot Circles and Non-Overlapping Value Badges
    for idx, r in enumerate(rows):
        cx = col_x(idx)
        cy = y_px(r["bw_gb_s"])
        svg.append(f'  <circle cx="{cx}" cy="{cy}" r="6.5" fill="#2563eb" stroke="#ffffff" stroke-width="2"/>')

        # Staggered badges to guarantee zero crowding
        badge_y = cy - 28 if idx % 2 == 0 else cy + 12
        badge_fill = "#1d4ed8" if r["bw_gb_s"] > 300 else "#334155"
        svg.append(f'  <rect x="{cx - 36}" y="{badge_y - 10}" width="72" height="18" class="val-badge" fill="{badge_fill}"/>')
        svg.append(f'  <text x="{cx}" y="{badge_y + 3}" class="val-text" text-anchor="middle">{r["bw_gb_s"]:.1f}</text>')

    svg.append('</svg>')

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
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "g": int(r["group_size"]),
                "leg_dram": float(r["legacy_dram_gflops"]),
                "c1_dram": float(r["coal1_tpl_dram_gflops"]),
                "c1_slc": float(r["coal1_tpl_slc_gflops"]),
                "c4_slc": float(r["coal4_tpl_slc_gflops"])
            })

    width = 1100
    height = 580
    pad_left = 110
    pad_right = 60
    pad_top = 105
    pad_bottom = 90
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    max_y = 1750.0

    def col_x(idx):
        return pad_left + (idx / (len(rows) - 1)) * plot_w

    def y_px(val):
        return height - pad_bottom - (val / max_y) * plot_h

    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        '  <style>',
        '    .bg { fill: #ffffff; }',
        '    .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 21px; font-weight: 700; fill: #0f172a; }',
        '    .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; fill: #475569; }',
        '    .axis-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; font-weight: 600; fill: #1e293b; }',
        '    .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4,4; }',
        '    .axis-line { stroke: #94a3b8; stroke-width: 1.5; }',
        '    .tick-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; }',
        '    .tick-sub { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; font-weight: 600; fill: #0284c7; }',
        '    .line-c4slc { stroke: #7c3aed; stroke-width: 3.5; fill: none; }',
        '    .line-c1dram { stroke: #2563eb; stroke-width: 3.5; fill: none; }',
        '    .line-leg { stroke: #dc2626; stroke-width: 2.5; stroke-dasharray: 6,4; fill: none; }',
        '  </style>',
        f'  <rect width="{width}" height="{height}" class="bg" rx="14" stroke="#cbd5e1" stroke-width="1.5"/>',
        f'  <text x="{pad_left}" y="42" class="title">Ablation 3: Group Granularity &amp; Architecture Scaling</text>',
        f'  <text x="{pad_left}" y="65" class="subtitle">Effective Compute (GFLOPS) across Group Sizes G ∈ [16, 256] | Matrix: 4608×4608</text>'
    ]

    # Legend (Escaped XML characters!)
    leg_x = width - pad_right - 580
    svg.append(f'  <line x1="{leg_x}" y1="36" x2="{leg_x+24}" y2="36" class="line-c4slc"/>')
    svg.append(f'  <text x="{leg_x+30}" y="40" font-family="-apple-system, sans-serif" font-size="11px" font-weight="600" fill="#7c3aed">Warm SLC (C4Tpl, 4-Row Register Tiled)</text>')

    svg.append(f'  <line x1="{leg_x}" y1="58" x2="{leg_x+24}" y2="58" class="line-c1dram"/>')
    svg.append(f'  <text x="{leg_x+30}" y="62" font-family="-apple-system, sans-serif" font-size="11px" font-weight="600" fill="#2563eb">Cold DRAM (MetalSlice C1Tpl) — Peak at NPOT G=72, 96</text>')

    svg.append(f'  <line x1="{leg_x+350}" y1="58" x2="{leg_x+374}" y2="58" class="line-leg"/>')
    svg.append(f'  <text x="{leg_x+380}" y="62" font-family="-apple-system, sans-serif" font-size="11px" font-weight="600" fill="#dc2626">Legacy Strided (Starves at G ≥ 128)</text>')

    # Gridlines Y
    for gflops_val in [600, 800, 1000, 1200, 1400, 1600]:
        py = y_px(gflops_val)
        svg.append(f'  <line x1="{pad_left}" y1="{py}" x2="{pad_left + plot_w}" y2="{py}" class="grid"/>')
        svg.append(f'  <text x="{pad_left - 14}" y="{py + 4}" font-family="-apple-system, sans-serif" font-size="11px" fill="#64748b" text-anchor="end">{gflops_val}</text>')

    # Axis Lines
    svg.append(f'  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{pad_left + plot_w}" y2="{height - pad_bottom}" class="axis-line"/>')
    svg.append(f'  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" class="axis-line"/>')

    # Axis Titles
    svg.append(f'  <text x="{pad_left + plot_w/2}" y="{height - 25}" class="axis-title" text-anchor="middle">Quantization Group Size (G) → Finer (Left) to Coarser (Right)</text>')
    svg.append(f'  <text x="32" y="{pad_top + plot_h/2}" class="axis-title" text-anchor="middle" transform="rotate(-90 32 {pad_top + plot_h/2})">Effective Compute (GFLOPS)</text>')

    # X-Axis Columns
    for idx, r in enumerate(rows):
        cx = col_x(idx)
        svg.append(f'  <line x1="{cx}" y1="{pad_top}" x2="{cx}" y2="{height - pad_bottom}" class="grid"/>')
        is_npot = r["g"] in [24, 48, 72, 96]
        sub = "NPOT" if is_npot else "POT"
        sub_color = "#0284c7" if is_npot else "#64748b"
        svg.append(f'  <text x="{cx}" y="{height - pad_bottom + 20}" class="tick-hdr" text-anchor="middle">G={r["g"]}</text>')
        svg.append(f'  <text x="{cx}" y="{height - pad_bottom + 34}" class="tick-sub" fill="{sub_color}" text-anchor="middle">{sub}</text>')

    # Polylines
    pts_c4 = [f'{col_x(i)},{y_px(r["c4_slc"])}' for i, r in enumerate(rows)]
    pts_c1dram = [f'{col_x(i)},{y_px(r["c1_dram"])}' for i, r in enumerate(rows)]
    pts_leg = [f'{col_x(i)},{y_px(r["leg_dram"])}' for i, r in enumerate(rows)]

    svg.append(f'  <polyline points="{" ".join(pts_c4)}" class="line-c4slc"/>')
    svg.append(f'  <polyline points="{" ".join(pts_c1dram)}" class="line-c1dram"/>')
    svg.append(f'  <polyline points="{" ".join(pts_leg)}" class="line-leg"/>')

    # Circles and Highlights
    for i, r in enumerate(rows):
        cx = col_x(i)

        # C4 Warm SLC
        py_c4 = y_px(r["c4_slc"])
        svg.append(f'  <circle cx="{cx}" cy="{py_c4}" r="5.5" fill="#7c3aed" stroke="#ffffff" stroke-width="2"/>')

        # C1 Cold DRAM
        py_c1 = y_px(r["c1_dram"])
        svg.append(f'  <circle cx="{cx}" cy="{py_c1}" r="6.5" fill="#2563eb" stroke="#ffffff" stroke-width="2"/>')

        # Highlight G=96 Peak in Cold DRAM
        if r["g"] == 96:
            svg.append(f'  <rect x="{cx - 75}" y="{py_c1 - 36}" width="150" height="24" rx="5" fill="#0284c7"/>')
            svg.append(f'  <text x="{cx}" y="{py_c1 - 20}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#ffffff" text-anchor="middle">★ Cold Peak: 1,003 GFLOPS</text>')
            svg.append(f'  <line x1="{cx}" y1="{py_c1}" x2="{cx}" y2="{py_c1 - 12}" stroke="#0284c7" stroke-width="1.5"/>')

        # Highlight G=72
        if r["g"] == 72:
            svg.append(f'  <rect x="{cx - 65}" y="{py_c1 + 14}" width="130" height="22" rx="4" fill="#0d9488"/>')
            svg.append(f'  <text x="{cx}" y="{py_c1 + 29}" font-family="-apple-system, sans-serif" font-size="10px" font-weight="700" fill="#ffffff" text-anchor="middle">NPOT G=72: 982 GFLOPS</text>')

        # Legacy Curve Point
        py_leg = y_px(r["leg_dram"])
        svg.append(f'  <circle cx="{cx}" cy="{py_leg}" r="5.5" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>')
        if r["g"] == 256:
            svg.append(f'  <rect x="{cx - 150}" y="{py_leg + 14}" width="160" height="24" rx="5" fill="#dc2626"/>')
            svg.append(f'  <text x="{cx - 70}" y="{py_leg + 30}" font-family="-apple-system, sans-serif" font-size="11px" font-weight="700" fill="#ffffff" text-anchor="middle">Legacy Starvation: -29.1%</text>')

    svg.append('</svg>')

    out_path = os.path.join(RESULTS_DIR, "group_granularity_scaling.svg")
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"[+] Wrote Group Granularity Chart: {out_path}")

def main():
    print("[*] Generating publication-grade SVG figures for MetalSlice...")
    generate_pareto_chart()
    generate_slc_cliff_chart()
    generate_group_scaling_chart()

    # XML Validation Check
    for f in ["throughput_energy_pareto.svg", "slc_eviction_curve.svg", "group_granularity_scaling.svg"]:
        p = os.path.join(RESULTS_DIR, f)
        ET.parse(p)
        print(f"[PASS] {f} passed XML syntax verification")
    print("[+] All charts successfully verified and ready!")

if __name__ == "__main__":
    main()
