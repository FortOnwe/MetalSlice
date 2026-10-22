# MetalSlice: Chronological Experiment & Senior Scientist Audit Log

**Author:** Fortune N. Onwe  
**Project:** MetalSlice: Eliminating SIMD Starvation and Hiding Non-Power-of-Two Integer Division Penalties on Apple Silicon Unified Memory  
**Repository:** [https://github.com/FortOnwe/MetalSlice](https://github.com/FortOnwe/MetalSlice)

---

## Phase 1: Initial Baseline & Identification of Five Methodological Confounders
- **Hardware Environment**: Apple Silicon M5 Pro, 18 CPU cores, 48 GB unified LPDDR5X memory.
- **Initial Observation**: Naive single-buffer host-timed Metal GEMV benchmarks suffered from five major experimental confounders:
  1. **System-Level Cache (SLC) Residency**: Single `MTLBuffer` benchmarks of 4096×4096 (8.5 MB in INT4) and 4608×4608 (10.8 MB in INT4) fit entirely inside the on-die SLC, inflating apparent memory bandwidth past 370–450 GB/s.
  2. **Host-Side Command Submission Overhead**: Timing `waitUntilCompleted` on single-dispatch `MTLCommandBuffer` submissions measured a ~130 µs driver latency floor rather than the ~30–45 µs GPU execution time.
  3. **Zero-Initialized Page Compression Confounder**: Allocating uninitialized or zero-filled buffers via `newBufferWithLength` allowed hardware zero-compression and lazy page faulting to skew DRAM bus power and bandwidth.
  4. **Group-Strided SIMD Thread Starvation**: Assigning 1 thread per quantization group (`for (uint g = tid; g < num_groups; g += 128)`) starved SIMD lanes whenever `num_groups = K / G < 128` (96 active threads at G=48, 64 at G=72, 36 at G=128, 18 at G=256), creating an artificial throughput collapse at G ≥ 128.
  5. **DVFS P-State Transients**: Short bursts without warmup or idle baseline subtraction conflated GPU clock ramp-up and background system power with active dequantization energy.

---

## Phase 2: Senior Scientist Refactor & 4-Architecture Kernel Suite (`src/kernels.metal`)
To isolate true hardware physics from software implementation artifacts, we re-engineered the MetalSlice compute suite from scratch:
1. **Deterministic Non-Zero PRNG Buffer Initialization (`fill_buffer_deterministic`)**:
   - Every buffer in the 32-buffer ring pool (329 MB to 5,504 MB) is populated using a 64-bit xorshift PRNG with NaN/Inf exponent bits masked, faulting all physical pages into DRAM before timing and ensuring realistic bit-toggle entropy on the LPDDR5X bus.
2. **Ground-Truth CPU FP64 Verification (`--mode verify`)**:
   - Added a double-precision (fp64) CPU reference verifier in `src/bench_runner.mm`. Verified all four precision families on K = 4608, N = 4608:
     - FP32: Max Error 4.03e-06, RMSE 2.89e-06 (PASS)
     - FP16: Max Error 5.86e-03, RMSE 3.21e-03 (PASS)
     - INT8 (G=64): Max Error 5.20e-03, RMSE 2.88e-03 (PASS)
     - INT4 (G=64): Max Error 9.85e-02, RMSE 5.56e-02 (PASS)
3. **Four-Architecture Kernel Ablation (`src/ablation_runner.mm`)**:
   - Built side-by-side kernels in `src/kernels.metal` to test across G ∈ {16, 24, 32, 48, 64, 72, 96, 128, 256}:
     - **Architecture 1 (`Leg`: `gemv_int4_group_strided`)**: Legacy 1-thread-per-group layout with barrier reduction.
     - **Architecture 2 (`C1Dyn`: `gemv_int4`)**: 100%-occupancy coalesced SIMDgroup layout (32 lanes × 8 weights = 256 weights = 128 packed bytes per SIMD step, 4 rows per 128-thread threadgroup) using runtime integer division (`elem_base / group_size`).
     - **Architecture 3 (`C1Tpl`: `gemv_int4_1row_G<G>`)**: Coalesced SIMDgroup layout specialized with compile-time template constant `<G>` (16, 24, 32, 48, 64, 72, 96, 128, 256).
     - **Architecture 4 (`C4Tpl`: `gemv_int4_4row_G<G>`)**: Coalesced SIMDgroup layout with 4-row register tiling (16 rows per threadgroup), amortizing vector loads and scale-index math across 4 output rows simultaneously.

---

## Phase 3: Execution of the Three Microarchitectural Ablation Studies

### Log 3.1: Ablation 1 — System-Level Cache (SLC) Evasion Curve (`results/ablation_ring_pool_slc.csv`)
- **Resolution of Kernel Consistency in Ablation 1**:
  - Refactored `runRingSizeAblation` in `src/ablation_runner.mm` to dispatch the coalesced `C1Tpl` kernel (`tgSize = 128`, `gridSize = (N+3)/4`), perfectly harmonizing it with `bench_runner.mm` and Ablation 3.
  - With a 16-dispatch warmup, `RingSize=1` (10.76 MB) clocked **28.90 µs (390.97 GB/s, 1,469.5 GFLOPS)**, `RingSize=2` (21.52 MB) clocked **31.63 µs (357.17 GB/s, 1,342.5 GFLOPS)**, and `RingSize=4` (43.03 MB) hit the **SLC eviction cliff at 42.19 µs (267.79 GB/s, 1,006.5 GFLOPS)**, stabilizing across `RingSize=6..64` (64.5–688.5 MB) at **42.37–43.57 µs (259.35–266.66 GB/s)**.

### Log 3.2: Ablation 2 — Command Buffer Batching (`results/ablation_batch_dispatch.csv`)
- Swept `BatchSize` ∈ {1, 2, 4, 8, 16, 32, 64} on 4608×4608 INT4-G64:
  - `BatchSize=1`: 177.04 µs host vs. 43.83 µs GPU (4.04x overhead).
  - `BatchSize=32`: 48.65 µs host vs. 43.35 µs GPU (1.12x overhead).
  - `BatchSize=64`: 46.72 µs host vs. 44.42 µs GPU (1.05x overhead).

### Log 3.3: Ablation 3 — Group Granularity & Architecture Scaling (`results/ablation_group_granularity.csv`)
- Confirmed why `Legacy Group-Strided` produced an artificial peak at G = 48–72 (43.74–44.17 µs, 961–970 GFLOPS) and collapsed at G = 128–256 (49.55–55.99 µs, 758–857 GFLOPS): SIMD lane starvation when `num_groups = 4608 / G < 128`.
- Confirmed that under `Coalesced 1-Row Template<G> (C1Tpl)`:
  - **In Warm-SLC**: Power-of-two group sizes (G ∈ {32, 64, 128, 256}) achieve **29.07–29.66 µs (1,431–1,460 GFLOPS)** (and **27.70–28.46 µs / 1,491–1,532 GFLOPS** under 4-row `C4Tpl`), beating non-power-of-two G = 72, 96 (29.62–30.40 µs / 1,396–1,433 GFLOPS) by **+5.2%** due to 1-cycle bitwise shift (`lsr`) vs. 3-cycle reciprocal multiply-high (`umulhi`).
  - **In Cold-DRAM**: The 2-cycle ALU difference is completely hidden by DRAM latency. Performance converges to a flat memory-bus roofline plateau across G ∈ {64, 72, 96, 128, 256} (**41.86–43.04 µs / 986–1,014 GFLOPS**), proving non-power-of-two group sizes incur zero memory-bus throughput penalty in Cold DRAM.

---

## Phase 4: Cross-Dimension Sweeps & Multi-Trial Power Telemetry (N=5)
- Executed full hardware-timestamped sweeps across four canonical LLM dimensions:
  1. 4096×4096 (`results/sweep_4096x4096.csv`)
  2. 4608×4608 (`results/sweep_gpu_timestamps_4608x4608.csv`)
  3. 11008×4096 (`results/sweep_11008x4096.csv`)
  4. 8192×8192 (`results/sweep_8192x8192.csv`) — reaches **1,049.7 GFLOPS (266.8 GB/s)** in Cold DRAM at INT4-G256 and **1,026.2 GFLOPS (272.8 GB/s)** at INT4-G64.
- Executed 5-trial randomized-order continuous power and thermal telemetry (`src/power_rerun_confirmation.py` → `results/ablation_multitrial_confirmation.csv`, `results/power_telemetry_rigorous.csv`, `results/power_telemetry_rigorous.json`, `results/energy_stats.json`) across 10 configurations (FP32, FP16, INT8-G64, and INT4 at G ∈ {32, 48, 64, 72, 96, 128, 256}) with per-run local idle baseline subtraction ($P_{\text{idle}} \in [0.99\text{ W}, 8.39\text{ W}]$) and 8.0s inter-run cooldown:
  - Welch's t-test between FP16 (1.1564 ± 0.0573 J) and INT4-G128 (1.2814 ± 0.0370 J): $t = -4.10, p = 0.0048$ ($df=6.8, N=5$).
  - Established FP16 (1.1564 ± 0.0573 J / 1k GEMV at 6,188.7 ± 45.3 GEMV/s, 10.74 ± 2.34 W active) as the Global Energy Minimum (memory read stalls holding the GPU in lower-frequency P-states), and INT4-G256 (20,839.5 ± 97.3 GEMV/s, 1.2623 ± 0.0469 J) / INT4-G128 (20,676.3 ± 66.2 GEMV/s, 1.2814 ± 0.0370 J) as the Pareto Throughput-Efficiency Knee (3.34x–3.37x faster than FP16, 44.2% lower energy than FP32).
  - Confirmed statistical parity across INT4 $G \in [64, 256]$ ($t = -0.72, p = 0.4956$ for G256 vs. G128; $t = 0.60, p = 0.5646$ for G72 vs. G128).

---

## Phase 5: Replicated Warm-SLC Ablation (N=20, Randomized Group Order)
- Executed Ablation 5 (`runWarmSlcRepeats` in `src/ablation_runner.mm` → `results/ablation_warm_slc_repeats.csv`, analyzed via `src/analyze_warm_slc.py` → `results/ablation_warm_slc_repeats_summary.md`):
  - In 1-row `C1Tpl`: Power-of-two group sizes average **1,463.1 GFLOPS** vs. non-PoT **1,412.7 GFLOPS** (+3.57% PoT advantage, 95% bootstrap CI [+3.56%, +3.59%], permutation $p < 0.001$), confirming the 1-cycle bitwise shift (`lsr`) vs. 3-cycle reciprocal multiply-high (`umulhi`) scale-indexing penalty.
  - In 4-row register-tiled `C4Tpl`: 4-row instruction interleaving amortizes the integer division penalty across rows, yielding **1,481.2 GFLOPS** for non-PoT vs. **1,473.1 GFLOPS** for PoT (-0.55%), peaking at non-PoT $G=72$ (**1,517.6 ± 0.7 GFLOPS**) and $G=256$ (**1,550.3 ± 1.0 GFLOPS**).
  - Formally refuted earlier unverified claims of a universal +5.2% PoT advantage across all architectures.

---

## Phase 6: Publication-Grade Figures & Interactive Dashboard Upgrades
- Rewrote `src/generate_plots.py` so that 100% of plot elements, ranges, error bars, and tables derive directly from CSV telemetry:
  1. `results/throughput_energy_pareto.svg`: Full 2D scatter with bi-directional error bars ($\pm 1\text{ SD}$ on GEMV/s and J/1k), non-overlapping callout pills, and embedded $N=5$ summary table.
  2. `results/slc_eviction_curve.svg`: Discrete categorical step layout with explicit ordinal axis subtitle, non-colliding eviction cliff callout badge ($y = 185\text{ GB/s}$), and pre-drawn zone fills.
  3. `results/group_granularity_scaling.svg`: Dual-regime plot displaying replicated $N=20$ `C4Tpl` and `C1Tpl` curves alongside Cold DRAM and Legacy Starvation, with zero-collision legend header ($y = 90$).
- All SVGs visually validated via native AppKit high-resolution rasterization (`scratch/render_svg.m`).
- Upgraded `docs/index.html` and `results/interactive_dashboard.html`:
  - Upgraded Pareto view to a true 2D scatter plot (Throughput vs. Net Energy) with dynamic tooltips and active package power subviews.
  - Embedded exact $N=5$ JSON telemetry and verified HTML structure with strict Python parser. Deployed for GitHub Pages (`https://fortonwe.github.io/MetalSlice/`).
