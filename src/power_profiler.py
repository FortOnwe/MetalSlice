#!/usr/bin/env python3
import subprocess
import json
import time
import sys
import os

MACMON_BIN = "/opt/homebrew/bin/macmon"
BENCH_BIN = "/Users/fo/Documents/Antigravity Test/metal_unified_memory_quant_study/bin/bench_runner"
RESULTS_DIR = "/Users/fo/Documents/Antigravity Test/metal_unified_memory_quant_study/results"

def measure_idle(seconds=5.0):
    print(f"[*] Measuring baseline idle power for {seconds} seconds...")
    proc = subprocess.Popen([MACMON_BIN, "pipe", "-i", "500"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    samples = []
    start = time.time()
    while time.time() - start < seconds:
        line = proc.stdout.readline()
        if line:
            try:
                data = json.loads(line.strip())
                samples.append(data)
            except Exception:
                pass
    proc.terminate()
    proc.wait()

    if not samples:
        return {"all_power": 0.6, "cpu_power": 0.2, "gpu_power": 0.1, "ram_power": 0.2, "temp": 36.0}

    # Discard first sample
    valid = samples[1:] if len(samples) > 2 else samples

    avg_all = sum(s.get("all_power", 0) for s in valid) / len(valid)
    avg_cpu = sum(s.get("cpu_power", 0) for s in valid) / len(valid)
    avg_gpu = sum(s.get("gpu_power", 0) for s in valid) / len(valid)
    avg_ram = sum(s.get("ram_power", 0) for s in valid) / len(valid)
    avg_sys = sum(s.get("sys_power", 0) for s in valid) / len(valid)
    avg_temp = sum(s.get("temp", {}).get("gpu_temp_avg", 36.0) for s in valid) / len(valid)

    print(f"    Baseline Idle (Stable): Package={avg_all:.2f}W (CPU={avg_cpu:.2f}W, GPU={avg_gpu:.2f}W, RAM={avg_ram:.2f}W, Sys={avg_sys:.2f}W, GPU Temp={avg_temp:.1f}C)")
    return {
        "all_power": avg_all,
        "cpu_power": avg_cpu,
        "gpu_power": avg_gpu,
        "ram_power": avg_ram,
        "sys_power": avg_sys,
        "gpu_temp_c": avg_temp
    }

def wait_for_cooldown(target_temp, timeout_sec=15.0):
    start = time.time()
    while time.time() - start < timeout_sec:
        proc = subprocess.Popen([MACMON_BIN, "pipe", "-i", "500"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        line = proc.stdout.readline()
        proc.terminate()
        proc.wait()
        if line:
            try:
                data = json.loads(line.strip())
                cur_temp = data.get("temp", {}).get("gpu_temp_avg", 0)
                if cur_temp <= target_temp + 2.0:
                    break
            except Exception:
                pass
        time.sleep(1.0)

def run_sustained_telemetry(precision, group_size, K=4608, N=4608, duration=6.0, idle_baseline=None):
    print(f"\n[*] Profiling {precision.upper()} (G={group_size}) for {duration}s continuous sustained burst (Cold DRAM Ring Pool)...")
    
    # Start macmon background collector
    proc_macmon = subprocess.Popen([MACMON_BIN, "pipe", "-i", "500"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Launch benchmark burst (using ring-size 32 to ensure DRAM bus saturation)
    bench_cmd = [
        BENCH_BIN,
        "--mode", "sustained",
        "--precision", precision,
        "--group-size", str(group_size),
        "--k", str(K),
        "--n", str(N),
        "--ring-size", "32",
        "--duration", str(duration)
    ]
    
    bench_proc = subprocess.Popen(bench_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    bench_stdout, bench_stderr = bench_proc.communicate()
    
    # Stop macmon after burst completes
    time.sleep(0.5)
    proc_macmon.terminate()
    macmon_stdout, _ = proc_macmon.communicate()

    samples = []
    for line in macmon_stdout.splitlines():
        if line.strip():
            try:
                data = json.loads(line.strip())
                samples.append(data)
            except Exception:
                pass

    # Parse benchmark stdout
    # SUSTAINED_BURST_COMPLETE: precision=int4 group_size=48 dispatches=135680 duration=6.0002 dispatches_per_sec=22612.5
    dispatches = 0
    actual_dur = duration
    dispatches_per_sec = 0.0
    for line in bench_stdout.splitlines():
        if "SUSTAINED_BURST_COMPLETE:" in line:
            parts = line.strip().split()
            for p in parts:
                if p.startswith("dispatches="): dispatches = int(p.split("=")[1])
                elif p.startswith("duration="): actual_dur = float(p.split("=")[1])
                elif p.startswith("dispatches_per_sec="): dispatches_per_sec = float(p.split("=")[1])

    # Methodological Control 5: Discard first 1.0 second ramp-up samples (first 2 samples at 500ms)
    valid_samples = samples[2:] if len(samples) > 4 else samples

    avg_all = sum(s.get("all_power", 0) for s in valid_samples) / len(valid_samples) if valid_samples else 0
    avg_cpu = sum(s.get("cpu_power", 0) for s in valid_samples) / len(valid_samples) if valid_samples else 0
    avg_gpu = sum(s.get("gpu_power", 0) for s in valid_samples) / len(valid_samples) if valid_samples else 0
    avg_ram = sum(s.get("ram_power", 0) for s in valid_samples) / len(valid_samples) if valid_samples else 0
    avg_sys = sum(s.get("sys_power", 0) for s in valid_samples) / len(valid_samples) if valid_samples else 0
    
    avg_temp_gpu = sum(s.get("temp", {}).get("gpu_temp_avg", 0) for s in valid_samples) / len(valid_samples) if valid_samples else 0
    avg_temp_cpu = sum(s.get("temp", {}).get("cpu_temp_avg", 0) for s in valid_samples) / len(valid_samples) if valid_samples else 0

    idle_all = idle_baseline.get("all_power", 0) if idle_baseline else 0
    net_power = max(0.0, avg_all - idle_all)
    
    # Net Energy (Joules) = Net Power * Duration
    total_net_joules = net_power * actual_dur
    joules_per_1k_dispatches = (total_net_joules / dispatches * 1000.0) if dispatches > 0 else 0
    
    # Parameters in matrix = N * K
    params = N * K
    mparams = params / 1e6
    joules_per_mparam = (total_net_joules / dispatches) / mparams if dispatches > 0 else 0
    
    # GFLOPs per GEMV = (2 * N * K) / 1e9
    gflops_per_gemv = (2.0 * N * K) / 1e9
    joules_per_gflop = (total_net_joules / dispatches) / gflops_per_gemv if dispatches > 0 else 0

    print(f"    Sustained: {dispatches} dispatches in {actual_dur:.2f}s ({dispatches_per_sec:.1f} GEMV/s)")
    print(f"    Steady-State Power: Active={avg_all:.2f}W | Net={net_power:.2f}W (GPU={avg_gpu:.2f}W, RAM={avg_ram:.2f}W, CPU={avg_cpu:.2f}W)")
    print(f"    Thermal: GPU={avg_temp_gpu:.1f}C (Baseline={idle_baseline['gpu_temp_c']:.1f}C, Delta=+{avg_temp_gpu - idle_baseline['gpu_temp_c']:.1f}C)")
    print(f"    Energy Metrics:")
    print(f"      - {joules_per_1k_dispatches:.4f} Joules / 1k GEMV")
    print(f"      - {joules_per_mparam*1e6:.4f} uJ / MParam-Op")
    print(f"      - {joules_per_gflop*1e3:.4f} mJ / GFLOP")

    # Cooldown pause
    time.sleep(3.0)
    wait_for_cooldown(idle_baseline["gpu_temp_c"], timeout_sec=8.0)

    return {
        "precision": precision,
        "group_size": group_size,
        "K": K,
        "N": N,
        "duration_s": actual_dur,
        "dispatches": dispatches,
        "dispatches_per_sec": dispatches_per_sec,
        "avg_active_power_w": avg_all,
        "avg_net_power_w": net_power,
        "avg_gpu_power_w": avg_gpu,
        "avg_ram_power_w": avg_ram,
        "avg_cpu_power_w": avg_cpu,
        "avg_sys_power_w": avg_sys,
        "avg_gpu_temp_c": avg_temp_gpu,
        "temp_delta_c": avg_temp_gpu - idle_baseline["gpu_temp_c"],
        "joules_per_1k_gemv": joules_per_1k_dispatches,
        "joules_per_mparam": joules_per_mparam,
        "joules_per_gflop": joules_per_gflop
    }

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    idle = measure_idle(seconds=5.0)

    configs = [
        ("fp32", 0),
        ("fp16", 0),
        ("int8", 32),
        ("int8", 48),
        ("int8", 64),
        ("int8", 128),
        ("int4", 16),
        ("int4", 32),
        ("int4", 48),
        ("int4", 64),
        ("int4", 96),
        ("int4", 128),
        ("int4", 256)
    ]

    all_records = []
    for prec, gs in configs:
        rec = run_sustained_telemetry(prec, gs, K=4608, N=4608, duration=6.0, idle_baseline=idle)
        all_records.append(rec)

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "power_telemetry_rigorous.json")
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "power_telemetry_rigorous.csv")
    with open(csv_path, "w") as f:
        headers = [
            "precision", "group_size", "K", "N", "duration_s", "dispatches", "dispatches_per_sec",
            "avg_active_power_w", "avg_net_power_w", "avg_gpu_power_w", "avg_ram_power_w",
            "avg_gpu_temp_c", "temp_delta_c", "joules_per_1k_gemv", "joules_per_mparam", "joules_per_gflop"
        ]
        f.write(",".join(headers) + "\n")
        for r in all_records:
            f.write(",".join(str(r[h]) for h in headers) + "\n")

    print(f"\n[+] Rigorous Telemetry Run Complete! Written to:\n  - {json_path}\n  - {csv_path}")

if __name__ == "__main__":
    main()
