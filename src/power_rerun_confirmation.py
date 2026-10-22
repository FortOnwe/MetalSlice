#!/usr/bin/env python3
import subprocess
import json
import time
import sys
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MACMON_BIN = os.environ.get("MACMON_BIN", "/opt/homebrew/bin/macmon")
BENCH_BIN = os.path.join(ROOT, "bin", "bench_runner")
RESULTS_DIR = os.path.join(ROOT, "results")
COOLDOWN_S = 8.0  # longer cooldown + randomized order to decouple config from thermal state

def measure_idle(seconds=4.0):
    proc = subprocess.Popen([MACMON_BIN, "pipe", "-i", "500"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    samples = []
    start = time.time()
    while time.time() - start < seconds:
        line = proc.stdout.readline()
        if line:
            try:
                samples.append(json.loads(line.strip()))
            except Exception:
                pass
    proc.terminate()
    proc.wait()
    valid = samples[1:] if len(samples) > 2 else samples
    avg_all = sum(s.get("all_power", 0) for s in valid) / len(valid)
    avg_temp = sum(s.get("temp", {}).get("gpu_temp_avg", 36.0) for s in valid) / len(valid)
    return avg_all, avg_temp

def run_single_trial(precision, group_size, trial_id, idle_power, order_idx=0):
    proc_macmon = subprocess.Popen([MACMON_BIN, "pipe", "-i", "500"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    bench_cmd = [
        BENCH_BIN,
        "--mode", "sustained",
        "--precision", precision,
        "--group-size", str(group_size),
        "--k", "4608",
        "--n", "4608",
        "--ring-size", "32",
        "--duration", "4.0"
    ]
    bench_proc = subprocess.Popen(bench_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    bench_stdout, _ = bench_proc.communicate()
    
    time.sleep(0.3)
    proc_macmon.terminate()
    macmon_stdout, _ = proc_macmon.communicate()

    samples = []
    for line in macmon_stdout.splitlines():
        if line.strip():
            try:
                samples.append(json.loads(line.strip()))
            except Exception:
                pass

    dispatches = 0
    actual_dur = 5.0
    for line in bench_stdout.splitlines():
        if "SUSTAINED_BURST_COMPLETE:" in line:
            parts = line.strip().split()
            for p in parts:
                if p.startswith("dispatches="): dispatches = int(p.split("=")[1])
                elif p.startswith("duration="): actual_dur = float(p.split("=")[1])

    valid = samples[2:] if len(samples) > 3 else samples
    avg_power = sum(s.get("all_power", 0) for s in valid) / len(valid) if valid else 0
    avg_ram_p = sum(s.get("ram_power", 0) for s in valid) / len(valid) if valid else 0
    avg_gpu_p = sum(s.get("gpu_power", 0) for s in valid) / len(valid) if valid else 0
    avg_gpu_t = sum(s.get("temp", {}).get("gpu_temp_avg", 0) for s in valid) / len(valid) if valid else 0
    
    net_power = max(0.0, avg_power - idle_power)
    total_net_joules = net_power * actual_dur
    joules_1k = (total_net_joules / dispatches * 1000.0) if dispatches > 0 else 0
    rate = dispatches / actual_dur

    return {
        "trial": trial_id,
        "order_idx": order_idx,
        "precision": precision,
        "group_size": group_size,
        "dispatches_per_sec": rate,
        "active_power_w": avg_power,
        "net_power_w": net_power,
        "ram_power_w": avg_ram_p,
        "gpu_power_w": avg_gpu_p,
        "gpu_temp_c": avg_gpu_t,
        "joules_per_1k": joules_1k
    }

def main():
    print("\n========================================================")
    print(" [ABLATION 4] Multi-Trial Rerun Confirmation (Seeds & Variance)")
    print("========================================================")
    
    idle_power, idle_temp = measure_idle(seconds=3.0)
    print(f"[*] Baseline Idle: {idle_power:.2f}W, Temp: {idle_temp:.1f}C")

    test_configs = [
        ("fp32", 0),
        ("fp16", 0),
        ("int8", 64),
        ("int4", 32),
        ("int4", 48),
        ("int4", 64),
        ("int4", 72),
        ("int4", 96),
        ("int4", 128),
        ("int4", 256)
    ]

    all_trials = []
    num_trials = 5
    random.seed(20261022)

    for trial in range(1, num_trials + 1):
        print(f"\n--- TRIAL {trial} of {num_trials} ---")
        order = test_configs[:]
        random.shuffle(order)
        for idx, (prec, gs) in enumerate(order):
            local_idle, _ = measure_idle(seconds=3.0)  # per-run idle baseline (background load drifts)
            res = run_single_trial(prec, gs, trial, local_idle, idx)
            res["idle_power_w"] = local_idle
            all_trials.append(res)
            print(f"  {prec.upper()}-G{gs}: {res['dispatches_per_sec']:.1f} GEMV/s | Active={res['active_power_w']:.2f}W (RAM={res['ram_power_w']:.2f}W) | {res['joules_per_1k']:.4f} J/1k | Temp={res['gpu_temp_c']:.1f}C")
            time.sleep(COOLDOWN_S) # cooldown

    # Statistical Aggregation
    csv_path = os.path.join(RESULTS_DIR, "ablation_multitrial_confirmation.csv")
    with open(csv_path, "w") as f:
        headers = ["trial", "order_idx", "idle_power_w", "precision", "group_size", "dispatches_per_sec", "active_power_w", "net_power_w", "ram_power_w", "gpu_power_w", "gpu_temp_c", "joules_per_1k"]
        f.write(",".join(headers) + "\n")
        for r in all_trials:
            f.write(",".join(str(r[h]) for h in headers) + "\n")

    print(f"\n[+] Multi-Trial Confirmation complete! Saved to: {csv_path}")

    # Summary table and rigorous summary export
    summary_records = []
    print("\n" + "="*105)
    print(f"{'Configuration':<14} | {'Throughput (GEMV/s)':<21} | {'Active Power (W)':<17} | {'RAM Power (W)':<14} | {'Joules / 1k GEMV':<18}")
    print("="*105)
    for prec, gs in test_configs:
        subset = [r for r in all_trials if r["precision"] == prec and r["group_size"] == gs]
        mean_tp = sum(r["dispatches_per_sec"] for r in subset) / len(subset)
        std_tp = (sum((r["dispatches_per_sec"] - mean_tp)**2 for r in subset) / max(1, len(subset) - 1))**0.5
        mean_pw = sum(r["active_power_w"] for r in subset) / len(subset)
        std_pw = (sum((r["active_power_w"] - mean_pw)**2 for r in subset) / max(1, len(subset) - 1))**0.5
        mean_npw = sum(r["net_power_w"] for r in subset) / len(subset)
        mean_ram = sum(r["ram_power_w"] for r in subset) / len(subset)
        std_ram = (sum((r["ram_power_w"] - mean_ram)**2 for r in subset) / max(1, len(subset) - 1))**0.5
        mean_gpu = sum(r["gpu_power_w"] for r in subset) / len(subset)
        mean_tmp = sum(r["gpu_temp_c"] for r in subset) / len(subset)
        mean_en = sum(r["joules_per_1k"] for r in subset) / len(subset)
        std_en = (sum((r["joules_per_1k"] - mean_en)**2 for r in subset) / max(1, len(subset) - 1))**0.5

        K, N = 4608, 4608
        mparams = (K * N) / 1e6
        gflops_per_gemv = (2.0 * K * N) / 1e9
        joules_per_gemv = mean_en / 1000.0
        summary_records.append({
            "precision": prec,
            "group_size": gs,
            "K": K,
            "N": N,
            "trials": num_trials,
            "dispatches_per_sec_mean": round(mean_tp, 2),
            "dispatches_per_sec_std": round(std_tp, 2),
            "avg_active_power_w_mean": round(mean_pw, 2),
            "avg_active_power_w_std": round(std_pw, 2),
            "avg_net_power_w_mean": round(mean_npw, 2),
            "avg_gpu_power_w_mean": round(mean_gpu, 2),
            "avg_ram_power_w_mean": round(mean_ram, 2),
            "avg_ram_power_w_std": round(std_ram, 2),
            "avg_gpu_temp_c_mean": round(mean_tmp, 1),
            "joules_per_1k_gemv_mean": round(mean_en, 4),
            "joules_per_1k_gemv_std": round(std_en, 4),
            "uj_per_mparam": round((joules_per_gemv / mparams) * 1e6, 4),
            "mj_per_gflop": round((joules_per_gemv / gflops_per_gemv) * 1e3, 4)
        })

        label = f"{prec}-G{gs}" if gs > 0 else prec
        print(f"{label:<14} | {mean_tp:8.1f} ± {std_tp:<7.1f} | {mean_pw:6.2f} ± {std_pw:<6.2f} | {mean_ram:5.2f} ± {std_ram:<5.2f} | {mean_en:7.4f} ± {std_en:<7.4f}")
    print("="*105)

    json_path = os.path.join(RESULTS_DIR, "power_telemetry_rigorous.json")
    with open(json_path, "w") as f:
        json.dump(summary_records, f, indent=2)

    rig_csv_path = os.path.join(RESULTS_DIR, "power_telemetry_rigorous.csv")
    with open(rig_csv_path, "w") as f:
        headers = list(summary_records[0].keys())
        f.write(",".join(headers) + "\n")
        for r in summary_records:
            f.write(",".join(str(r[h]) for h in headers) + "\n")

if __name__ == "__main__":
    main()
