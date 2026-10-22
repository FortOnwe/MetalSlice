CXX = clang++
CXXFLAGS = -O3 -std=c++17 -framework Foundation -framework Metal

all: bin/bench_runner bin/ablation_runner

bin/bench_runner: src/bench_runner.mm src/kernels.metal
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) src/bench_runner.mm -o bin/bench_runner

bin/ablation_runner: src/ablation_runner.mm src/kernels.metal
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) src/ablation_runner.mm -o bin/ablation_runner

verify: bin/bench_runner
	./bin/bench_runner --mode verify --k 4608 --n 4608 --group-size 64

sweep: bin/bench_runner
	@mkdir -p results
	./bin/bench_runner --mode sweep --k 4608 --n 4608 --iters 512 --csv results/sweep_gpu_timestamps_4608x4608.csv
	./bin/bench_runner --mode sweep --k 4096 --n 4096 --iters 512 --csv results/sweep_4096x4096.csv
	./bin/bench_runner --mode sweep --k 11008 --n 4096 --iters 512 --csv results/sweep_11008x4096.csv
	./bin/bench_runner --mode sweep --k 8192 --n 8192 --iters 256 --ring-size 16 --csv results/sweep_8192x8192.csv

ablation: bin/ablation_runner
	@mkdir -p results
	./bin/ablation_runner

warm-repeats: bin/ablation_runner
	@mkdir -p results
	./bin/ablation_runner warm-repeats
	python3 src/analyze_warm_slc.py

power: bin/bench_runner
	python3 src/power_rerun_confirmation.py
	python3 src/energy_stats.py
	python3 src/generate_plots.py

clean:
	rm -rf bin/bench_runner bin/ablation_runner
