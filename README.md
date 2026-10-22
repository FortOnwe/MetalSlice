# MetalSlice: Eliminating SIMD Starvation and Hiding Non-Power-of-Two Integer Division Penalties on Apple Silicon Unified Memory

**Author:** Fortune N. Onwe  
**Repository:** [https://github.com/FortOnwe/MetalSlice](https://github.com/FortOnwe/MetalSlice)  
**Interactive Benchmark Dashboard:** [https://fortonwe.github.io/MetalSlice/](https://fortonwe.github.io/MetalSlice/) *(or open `docs/index.html` locally)*  
**Testbed Hardware:** Apple MacBook Pro (Apple M5 Pro, 18 CPU cores, 48 GB unified LPDDR5X memory, macOS 27.0.1)  
**Framework:** Native Metal Shading Language (MSL) & Objective-C++ (`-O3 -std=c++17 -framework Foundation -framework Metal`)

---

## 1. Research Objective & Core Discoveries

**MetalSlice** investigates the microarchitectural, System-Level Cache (SLC), and Dynamic Voltage and Frequency Scaling (DVFS) laws governing **block-quantized General Matrix-Vector Multiplication (GEMV, batch size M=1)** on Apple Silicon unified memory using custom zero-dependency Metal compute kernels and hardware-level telemetry.

### Primary Scientific Findings
1. **System-Level Cache (SLC) Isolation & the ~32–43 MB Capacity Boundary**:
   - Using an explicit buffer ring-pool ablation (`RingSize` 1 to 64), MetalSlice maps the exact working-set boundary where the GPU transitions from **on-die SRAM hit bandwidth (`357.17–390.97 GB/s`, `1,342.5–1,469.5 GFLOPS` at ≤ 21.5 MB)** to **cold LPDDR5X DRAM saturation (`259.35–267.79 GB/s` at ≥ 43.0 MB)**.
   - Standard single-buffer benchmarks of 4-bit LLM layers (10.8 MB for 4608×4608) measure on-die SLC cache hits rather than unified memory bandwidth, inflating apparent throughput by **+46.0%**.
2. **Eliminating Large-Group SIMD Starvation & Hiding Non-Power-of-Two Integer Division Penalties**:
   - In legacy group-strided kernels (`gemv_int4_group_strided`, 1 thread per group), throughput creates a false "peak" at G = 72 because larger group sizes suffer catastrophic SIMD lane starvation when `num_groups = K / G < 128` (72%–86% idle lanes at G ≥ 128, collapsing throughput to `49.59–57.82 µs` / `734–856 GFLOPS`).
   - By introducing **MetalSlice's 100%-Occupancy Coalesced SIMDgroup Kernels** (`32 lanes × 8 elements = 128 packed bytes` = 1 exact Apple Silicon cache line per SIMD step, 4 rows per 128-thread threadgroup with compile-time template specialization `<G>`), thread starvation is eliminated:
     - **In Warm-SLC (Compute / SRAM Bound)**: Under 1-row `C1Tpl`, power-of-two group sizes ($G \in \{16, 32, 64, 128, 256\}$) achieve a mean throughput of **`1,463.1 GFLOPS`** versus **`1,412.7 GFLOPS`** for non-power-of-two sizes ($G \in \{24, 48, 72, 96\}$)—a statistically significant **+3.57% PoT advantage** ($p < 0.001$, 95% bootstrap CI $[+3.56\%, +3.59\%]$, $N=20$ randomized trials) driven by the 1-cycle bitwise shift (`lsr`) versus 3-cycle reciprocal multiply-high (`umulhi`) scale-index calculation. However, under 4-row register tiling (`C4Tpl`), instruction-level interleaving across rows amortizes arithmetic overhead: non-PoT group sizes achieve **`1,481.2 GFLOPS`** versus PoT **`1,473.1 GFLOPS`** ($-0.55\%$), peaking at non-PoT $G=72$ (**`1,517.6 ± 0.7 GFLOPS`**) and $G=256$ (**`1,550.3 ± 1.0 GFLOPS`**).
     - **In Cold-DRAM (Unified Memory Bus Bound)**: The integer arithmetic difference is **100% hidden behind LPDDR5X DRAM latency (`250–284 GB/s`)**. Throughput forms a flat bandwidth-bound plateau across **G ∈ {64, 72, 96, 128, 256} (`41.86–43.04 µs` / `986–1,014 GFLOPS` in sweeps, `20,235–20,840 GEMV/s` sustained)** once scale-factor byte overhead falls below 6.25%. This proves fine-grained non-power-of-two quantization blocks run at **zero throughput cost** in Cold DRAM.
3. **The DVFS Power Wall & Pareto Frontier (FP16 Energy Minimum vs. INT4 Speedup)**:
   - Multi-trial hardware telemetry ($N=5$ randomized-order independent trials with per-run idle baseline subtraction, Welch's t-test: $t = -4.10, p = 0.0048$) demonstrates that reducing memory traffic does not linearly reduce Joules per operation:
     - **Global Minimum Energy (`FP16`)**: `FP16` achieves **`1.1564 ± 0.0573 J / 1k GEMV`** at `6,188.7 ± 45.3 GEMV/s` ($10.74 \pm 2.34\text{ W}$ active package power, $7.16\text{ W}$ net power) because memory-stalled execution prevents the GPU from entering high-frequency compute boost states.
     - **Pareto Throughput Knee (`INT4-G128` & `INT4-G256`)**: Nibble-unpacking ALUs drive the GPU into its full ~30.5–31.2 W compute state, delivering **`20,676.3 ± 66.2 GEMV/s` (`INT4-G128`)** and **`20,839.5 ± 97.3 GEMV/s` (`INT4-G256`)** with net energy of **`1.2814 ± 0.0370 J`** and **`1.2623 ± 0.0469 J / 1k GEMV`**. This achieves a **3.34x–3.37x speedup over FP16** (6.54x–6.60x over FP32) while delivering a **44.2% net energy reduction compared to FP32** (`1.2623 J` vs. `2.2622 ± 0.2300 J / 1k GEMV`). Within INT4 $G \in [64, 256]$ ($1.262–1.294\text{ J}$), energy differences are statistically indistinguishable ($p \ge 0.50$).

---

## 2. Methodological Controls (Zero-Artifact Benchmarking)
1. **Cold-DRAM Ring-Pool Rotation (SLC Isolation)**: Allocates 32 independent weight matrices (`329 MB` to `2,592 MB` for 4608×4608) and rotates buffer bindings every dispatch (`buf_idx = iter % 32`) to defeat the ~32–43 MB GPU SLC.
2. **Deterministic Non-Zero Bit-Toggle Entropy**: All matrix buffers are initialized via a deterministic 64-bit xorshift PRNG (masking NaN/Inf exponent bits) prior to timing, faulting every physical page into unified DRAM and ensuring realistic bit-toggle power on the LPDDR5X data bus.
3. **Hardware GPU Timestamps & 32-Dispatch Batching**: Measures exact GPU command-processor execution via `cmdBuf.GPUEndTime - cmdBuf.GPUStartTime` across 32 batched dispatches per `MTLCommandBuffer`, eliminating the **3.79x (129.75 µs)** host submission overhead.
4. **Compile-Time Template Specialization (`<G>`)**: Evaluates both dynamic uniform division and compile-time template constants (`gemv_int4_1row_G<G>` and `gemv_int4_4row_G<G>`) across power-of-two and non-power-of-two G ∈ {16, 24, 32, 48, 64, 72, 96, 128, 256}.
5. **Ground-Truth Numerical Verification (`--mode verify`)**: Validates all `FP32`, `FP16`, `INT8`, and `INT4` Metal kernels against a double-precision (fp64) CPU reference before benchmarking.

---

## 3. Multi-Trial Power, Throughput & Energy Results (4608×4608, Cold DRAM)
*Matrix: K=4608, N=4608 (21.23M parameters) | Ring Pool: 32 buffers (≥ 329 MB) | N=5 Randomized-Order Independent Trials (4.0s sustained burst/trial, 8.0s inter-run cooldown, first 1.0s discarded, per-run local idle baseline subtracted)*

| Configuration | Scale Byte Overhead | Sustained Throughput (GEMV/s) | Active Package Power (W) | LPDDR5X RAM Power (W) | Net Energy (Joules / 1k GEMV) | Speedup vs. FP32 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 0.00% | 3,159.3 ± 28.0 | 11.23 ± 2.38 | 9.83 ± 0.87 | 2.2622 ± 0.2300 | 1.00x |
| **FP16 (Energy Min)** | 0.00% | 6,188.7 ± 45.3 | 10.74 ± 2.34 | 10.14 ± 1.45 | **1.1564 ± 0.0573** | 1.96x |
| **INT8 (G=64)** | 3.13% | 11,275.5 ± 92.5 | 22.39 ± 2.28 | 10.04 ± 0.08 | 1.5982 ± 0.0452 | 3.57x |
| **INT4 (G=32)** | 12.50% | 19,369.3 ± 71.7 | 29.75 ± 1.50 | 10.96 ± 0.09 | 1.2859 ± 0.0644 | 6.13x |
| **INT4 (G=48, NPOT)** | 8.33% | 19,887.2 ± 57.0 | 30.17 ± 2.23 | 10.63 ± 0.11 | 1.3253 ± 0.0357 | 6.29x |
| **INT4 (G=64)** | 6.25% | 20,235.1 ± 82.1 | 30.84 ± 2.32 | 10.64 ± 0.13 | 1.2827 ± 0.0382 | 6.40x |
| **INT4 (G=72, NPOT)** | 5.56% | 20,367.1 ± 99.6 | 31.24 ± 2.74 | 9.70 ± 1.47 | 1.2937 ± 0.0266 | 6.45x |
| **INT4 (G=96, NPOT)** | 4.17% | 20,560.9 ± 61.7 | 30.83 ± 2.08 | 10.48 ± 0.10 | 1.3152 ± 0.0457 | 6.51x |
| **INT4 (G=128)** | 3.13% | 20,676.3 ± 66.2 | 30.50 ± 2.54 | 10.36 ± 0.08 | 1.2814 ± 0.0370 | 6.54x |
| **INT4 (G=256, Peak)** | 1.56% | **20,839.5 ± 97.3** | 31.24 ± 2.84 | 10.31 ± 0.11 | **1.2623 ± 0.0469** | **6.60x** |

![Throughput vs Energy Pareto Frontier](results/throughput_energy_pareto.svg)

---

## 4. Microarchitectural Ablation Studies

### Ablation 1: System-Level Cache (SLC) Evasion Curve (INT4-G64, 4608×4608)
*Varying RingSize from 1 (10.8 MB) to 64 (688.5 MB) using Coalesced `C1Tpl` with 16-dispatch GPU warmup (`results/ablation_ring_pool_slc.csv`):*

| Ring Pool Buffers | Total Working Set (MB) | GPU Median Latency (µs) | Effective Bandwidth (GB/s) | Effective Compute (GFLOPS) | Memory Hierarchy Regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RingSize = 1** | 10.76 MB | **28.90 µs** | **390.97 GB/s** | **1,469.5 GFLOPS** | **100% On-Die SLC Hit** |
| **RingSize = 2** | 21.52 MB | **31.63 µs** | **357.17 GB/s** | **1,342.5 GFLOPS** | **On-Die SLC Hit (Near Capacity)** |
| **RingSize = 4** | 43.03 MB | 42.19 µs | 267.79 GB/s | 1,006.5 GFLOPS | **SLC Eviction Cliff (→ Cold DRAM)** |
| **RingSize = 8** | 86.06 MB | 42.58 µs | 265.34 GB/s | 997.3 GFLOPS | Cold LPDDR5X DRAM |
| **RingSize = 16** | 172.13 MB | 42.54 µs | 265.59 GB/s | 998.3 GFLOPS | Cold LPDDR5X DRAM |
| **RingSize = 32** | 344.25 MB | 43.30 µs | 260.94 GB/s | 980.8 GFLOPS | Cold LPDDR5X DRAM |
| **RingSize = 64** | 688.50 MB | 43.57 µs | 259.35 GB/s | 974.8 GFLOPS | Cold LPDDR5X DRAM |

![GPU System-Level Cache Eviction Curve](results/slc_eviction_curve.svg)

### Ablation 2: Command Buffer Dispatch Batching (INT4-G64, 4608×4608)
*Isolating host driver submission latency vs. GPU hardware execution (`results/ablation_batch_dispatch.csv`):*
- At `BatchSize = 1`, host wall-clock latency is **177.04 µs** while true GPU execution is **43.83 µs**—a **4.04x (133.21 µs)** host driver submission tax.
- At `BatchSize = 32` (48.65 µs host vs. 43.35 µs GPU, 1.12x) and `BatchSize = 64` (46.72 µs host vs. 44.42 µs GPU, 1.05x), host submission overhead is completely amortized.

### Ablation 3: 4-Architecture Kernel Comparison Across Group Granularities (4608×4608)
*Comparing (1) `Leg`: Legacy 1-Thread-Per-Group, (2) `C1Dyn`: Coalesced 1-Row Dynamic Division, (3) `C1Tpl`: Coalesced 1-Row Compile-Time Template `<G>`, and (4) `C4Tpl`: Coalesced 4-Row Compile-Time Template `<G>` (`results/ablation_group_granularity.csv`):*

| Group Size (G) | Scale Overhead | Leg Cold DRAM (µs / GFLOPS) | C1Dyn Cold DRAM (µs / GFLOPS) | C1Tpl Cold DRAM (µs / GFLOPS) | C1Tpl Warm SLC (µs / GFLOPS) | C4Tpl Warm SLC (µs / GFLOPS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **G = 16** | 25.00% | 52.37 µs (810) | 50.28 µs (844) | 50.66 µs (838) | 30.31 µs (1,401) | 31.60 µs (1,344) |
| **G = 24 (NPOT)** | 16.67% | 48.12 µs (882) | 48.04 µs (883) | 46.89 µs (905) | 29.72 µs (1,428) | 28.96 µs (1,466) |
| **G = 32** | 12.50% | 45.46 µs (934) | 46.16 µs (919) | 44.49 µs (954) | **29.30 µs (1,449)** | 29.81 µs (1,424) |
| **G = 48 (NPOT)** | 8.33% | **44.17 µs (961)** | 45.79 µs (927) | 43.78 µs (970) | 29.63 µs (1,433) | 28.14 µs (1,509) |
| **G = 64** | 6.25% | **43.73 µs (971)** | 45.00 µs (943) | **43.04 µs (986)** | **29.47 µs (1,440)** | **28.54 µs (1,488)** |
| **G = 72 (NPOT)** | 5.56% | 43.74 µs (970) | 44.36 µs (957) | **42.16 µs (1,007)** | 30.40 µs (1,396) | 28.35 µs (1,498) |
| **G = 96 (NPOT)** | 4.17% | **44.61 µs (951)** | 43.83 µs (968) | **41.95 µs (1,012)** | 29.62 µs (1,433) | **27.86 µs (1,524)** |
| **G = 128** | 3.13% | 49.55 µs (857) | 44.74 µs (949) | **42.38 µs (1,002)** | **29.07 µs (1,460)** | **27.70 µs (1,532)** |
| **G = 256** | 1.56% | 55.99 µs (758) | 42.83 µs (991) | **41.86 µs (1,014)** | **29.66 µs (1,431)** | **28.46 µs (1,491)** |

![Group Granularity and Architecture Scaling](results/group_granularity_scaling.svg)

> [!TIP]
> **Live Interactive Benchmark Suite:** Explore parameter sweeps, real-time SIMD lane starvation collapse, SLC eviction curves, and DVFS Pareto scaling directly in your browser by opening [`docs/index.html`](docs/index.html) or viewing the [live GitHub Pages deployment](https://fortonwe.github.io/MetalSlice/).

---

## 5. Repository Structure & Reproduction Commands

```text
MetalSlice/
├── README.md                  # Executive summary, verified tables, and reproduction guide
├── analysis.md                # Deep-dive microarchitectural & DVFS physics analysis
├── EXPERIMENT_LOG.md          # Complete chronological engineering & audit log
├── DISCOVERY_CLAIM.md         # Formal scientific claim, confidence, and falsification criteria
├── CITATION.cff               # Formal academic citation metadata (Author: Fortune N. Onwe)
├── LICENSE                    # MIT License
├── Makefile                   # One-command build & benchmark automation
├── docs/
│   └── index.html             # Live interactive multi-experiment benchmark suite (GitHub Pages)
├── src/
│   ├── kernels.metal          # Coalesced SIMDgroup FP32/FP16/INT8/INT4 + Template<G> MSL shaders
│   ├── bench_runner.mm        # Metal hardware-timestamp benchmark & CPU fp64 verifier
│   ├── ablation_runner.mm     # Microarchitectural ablation suite (SLC, Batching, 4-Kernel, Warm-SLC repeats)
│   ├── power_rerun_confirmation.py # Multi-trial (N=5) randomized-order macmon power & energy profiler
│   ├── power_profiler.py      # Extended single-pass sustained power profiler
│   ├── analyze_warm_slc.py    # Statistical analysis & bootstrap CI on warm-SLC replications
│   ├── energy_stats.py        # Welch's t-test statistical engine (no scipy dependency)
│   └── generate_plots.py      # Pure-Python SVG generator for Pareto frontier charts
└── results/
    ├── interactive_benchmark.html   # Standalone interactive kernel throughput widget
    ├── interactive_dashboard.html   # Full multi-experiment interactive dashboard
    ├── sweep_gpu_timestamps_4608x4608.csv
    ├── sweep_4096x4096.csv
    ├── sweep_11008x4096.csv
    ├── sweep_8192x8192.csv
    ├── ablation_ring_pool_slc.csv
    ├── ablation_batch_dispatch.csv
    ├── ablation_group_granularity.csv
    ├── ablation_warm_slc_repeats.csv
    ├── ablation_warm_slc_repeats_summary.md
    ├── ablation_multitrial_confirmation.csv
    ├── power_telemetry_rigorous.csv
    ├── power_telemetry_rigorous.json
    ├── energy_stats.json
    ├── throughput_energy_pareto.svg
    ├── slc_eviction_curve.svg
    └── group_granularity_scaling.svg
```

### Reproducing the Study from Scratch
```bash
# 1. Build all binaries and run ground-truth FP64 verification
make clean
make all
make verify

# 2. Run GPU hardware-timestamped sweeps, 3-part ablations, and multi-trial power telemetry
make sweep
make ablation
make power
```

---

## 6. Citation

```bibtex
@software{onwe2026metalslice,
  author       = {Onwe, Fortune N.},
  title        = {{MetalSlice: Eliminating SIMD Starvation and Hiding Non-Power-of-Two Integer Division Penalties on Apple Silicon Unified Memory}},
  year         = {2026},
  month        = oct,
  url          = {https://github.com/FortOnwe/MetalSlice},
  platform     = {Apple Silicon (Metal Shading Language)}
}
```
