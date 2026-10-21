#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <map>
#include <cstring>

static inline void fill_buffer_deterministic(void* ptr, size_t bytes, uint64_t seed) {
    uint64_t* p64 = reinterpret_cast<uint64_t*>(ptr);
    size_t n64 = bytes / sizeof(uint64_t);
    uint64_t state = seed ^ 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < n64; ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        p64[i] = (state * 0x2545F4914F6CDD1DULL) & 0x3BFF3BFF3BFF3BFFULL;
    }
}

class AblationSuite {
private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pso_fp16;
    id<MTLComputePipelineState> pso_int4_dyn;
    id<MTLComputePipelineState> pso_int4_legacy;
    std::map<uint32_t, id<MTLComputePipelineState>> pso_int4_1row_tpl;
    std::map<uint32_t, id<MTLComputePipelineState>> pso_int4_4row_tpl;
    mach_timebase_info_data_t timebase;

public:
    AblationSuite() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("No Metal device found");
        commandQueue = [device newCommandQueue];
        mach_timebase_info(&timebase);
        loadKernels();
    }

    void loadKernels() {
        NSString *metalPath = @"/Users/fo/Documents/Antigravity Test/metal_unified_memory_quant_study/src/kernels.metal";
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&error];
        if (error || !source) throw std::runtime_error("Failed to load kernels.metal");

        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        library = [device newLibraryWithSource:source options:options error:&error];
        if (error || !library) throw std::runtime_error("Metal compilation failed");

        pso_fp16        = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gemv_fp16"] error:&error];
        pso_int4_dyn    = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gemv_int4"] error:&error];
        pso_int4_legacy = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gemv_int4_group_strided"] error:&error];

        std::vector<uint32_t> groups = {16, 24, 32, 48, 64, 72, 96, 128, 256};
        for (uint32_t g : groups) {
            NSString* n1 = [NSString stringWithFormat:@"gemv_int4_1row_G%u", g];
            NSString* n4 = [NSString stringWithFormat:@"gemv_int4_4row_G%u", g];
            pso_int4_1row_tpl[g] = [device newComputePipelineStateWithFunction:[library newFunctionWithName:n1] error:&error];
            pso_int4_4row_tpl[g] = [device newComputePipelineStateWithFunction:[library newFunctionWithName:n4] error:&error];
        }
    }

    double ticksToUs(uint64_t ticks) {
        return (double)ticks * (double)timebase.numer / ((double)timebase.denom * 1000.0);
    }

    double measureKernelLatencyUs(
        id<MTLComputePipelineState> pso,
        MTLSize gridSize,
        MTLSize tgSize,
        const std::vector<id<MTLBuffer>>& weight_ring,
        const std::vector<id<MTLBuffer>>& scale_ring,
        id<MTLBuffer> buf_X,
        id<MTLBuffer> buf_Y,
        uint32_t K,
        uint32_t N,
        uint32_t group_size,
        bool pass_group_arg,
        bool cold_dram,
        uint32_t batch_size = 32,
        uint32_t num_batches = 30)
    {
        uint32_t ring_size = (uint32_t)weight_ring.size();
        // Warmup
        {
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            for (uint32_t d = 0; d < 16; ++d) {
                uint32_t buf_idx = cold_dram ? (d % ring_size) : 0;
                [enc setBuffer:weight_ring[buf_idx] offset:0 atIndex:0];
                [enc setBuffer:scale_ring[buf_idx] offset:0 atIndex:1];
                [enc setBuffer:buf_X offset:0 atIndex:2];
                [enc setBuffer:buf_Y offset:0 atIndex:3];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
                if (pass_group_arg) {
                    [enc setBytes:&group_size length:sizeof(uint32_t) atIndex:6];
                }
                [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        std::vector<double> latencies;
        uint32_t ring_cursor = 0;

        for (uint32_t b = 0; b < num_batches; ++b) {
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];

            for (uint32_t d = 0; d < batch_size; ++d) {
                uint32_t buf_idx = cold_dram ? ((ring_cursor + d) % ring_size) : 0;
                [enc setBuffer:weight_ring[buf_idx] offset:0 atIndex:0];
                [enc setBuffer:scale_ring[buf_idx] offset:0 atIndex:1];
                [enc setBuffer:buf_X offset:0 atIndex:2];
                [enc setBuffer:buf_Y offset:0 atIndex:3];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
                if (pass_group_arg) {
                    [enc setBytes:&group_size length:sizeof(uint32_t) atIndex:6];
                }
                [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            ring_cursor = (ring_cursor + batch_size) % ring_size;
            double batch_s = [cmd GPUEndTime] - [cmd GPUStartTime];
            latencies.push_back((batch_s / (double)batch_size) * 1e6);
        }

        std::sort(latencies.begin(), latencies.end());
        return latencies[latencies.size() / 2];
    }

    // Ablation 1: Ring Buffer Pool Size (SLC Evasion Curve)
    void runRingSizeAblation(const std::string& csv_path) {
        std::cout << "\n========================================================" << std::endl;
        std::cout << " [ABLATION 1] Ring Buffer Pool Size (SLC Transition Curve)" << std::endl;
        std::cout << " Matrix: 4608x4608 | Precision: INT4 (G=64) | BatchSize: 32" << std::endl;
        std::cout << "========================================================" << std::endl;

        const uint32_t K = 4608, N = 4608, group_size = 64;
        const uint32_t num_groups = K / group_size;
        const size_t matrix_bytes = (size_t)N * (K / 2) + (size_t)N * num_groups * 2;
        const double total_bytes = (double)matrix_bytes + (double)(K + N) * 2.0;

        std::vector<uint32_t> ring_sizes = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64};

        std::ofstream csv(csv_path);
        csv << "ring_size,total_alloc_mb,median_gpu_us,effective_bw_gb_s,gflops\n";

        MTLSize gridSize = MTLSizeMake((N + 3) / 4, 1, 1);
        MTLSize tgSize   = MTLSizeMake(128, 1, 1);
        id<MTLComputePipelineState> pso = pso_int4_1row_tpl[group_size];

        for (uint32_t ring_size : ring_sizes) {
            double total_mb = (double)(matrix_bytes * ring_size) / (1024.0 * 1024.0);

            std::vector<id<MTLBuffer>> weight_ring(ring_size);
            std::vector<id<MTLBuffer>> scale_ring(ring_size);
            for (uint32_t b = 0; b < ring_size; ++b) {
                weight_ring[b] = [device newBufferWithLength:N * (K / 2) options:MTLResourceStorageModeShared];
                scale_ring[b]  = [device newBufferWithLength:N * num_groups * 2 options:MTLResourceStorageModeShared];
                fill_buffer_deterministic([weight_ring[b] contents], N * (K / 2), b + 111);
                fill_buffer_deterministic([scale_ring[b] contents], N * num_groups * 2, b + 222);
            }

            id<MTLBuffer> buf_X = [device newBufferWithLength:K * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_Y = [device newBufferWithLength:N * 2 options:MTLResourceStorageModeShared];
            fill_buffer_deterministic([buf_X contents], K * 2, 333);
            fill_buffer_deterministic([buf_Y contents], N * 2, 444);

            double med_us = measureKernelLatencyUs(
                pso, gridSize, tgSize, weight_ring, scale_ring, buf_X, buf_Y,
                K, N, group_size, false, true, 32, 30);

            double bw = (total_bytes / (med_us * 1e-6)) / 1e9;
            double gflops = (2.0 * (double)N * (double)K / (med_us * 1e-6)) / 1e9;

            std::cout << "RingSize: " << std::setw(2) << ring_size
                      << " (" << std::setw(6) << std::fixed << std::setprecision(1) << total_mb << " MB) -> "
                      << "Median: " << std::setw(6) << std::setprecision(2) << med_us << " us | "
                      << "BW: " << std::setw(6) << bw << " GB/s | "
                      << "GFLOPS: " << gflops << std::endl;

            csv << ring_size << "," << total_mb << "," << med_us << "," << bw << "," << gflops << "\n";
        }
    }

    // Ablation 2: Command Batch Size (Host vs GPU Timing Overhead)
    void runBatchSizeAblation(const std::string& csv_path) {
        std::cout << "\n========================================================" << std::endl;
        std::cout << " [ABLATION 2] Command Batch Size (Host vs Hardware GPU Time)" << std::endl;
        std::cout << " Matrix: 4608x4608 | Ring: 32 buffers | Precision: INT4-G64" << std::endl;
        std::cout << "========================================================" << std::endl;

        const uint32_t K = 4608, N = 4608, group_size = 64, ring_size = 32;
        const uint32_t num_groups = K / group_size;

        std::vector<id<MTLBuffer>> weight_ring(ring_size);
        std::vector<id<MTLBuffer>> scale_ring(ring_size);
        for (uint32_t b = 0; b < ring_size; ++b) {
            weight_ring[b] = [device newBufferWithLength:N * (K / 2) options:MTLResourceStorageModeShared];
            scale_ring[b]  = [device newBufferWithLength:N * num_groups * 2 options:MTLResourceStorageModeShared];
            fill_buffer_deterministic([weight_ring[b] contents], N * (K / 2), b + 111);
            fill_buffer_deterministic([scale_ring[b] contents], N * num_groups * 2, b + 222);
        }

        id<MTLBuffer> buf_X = [device newBufferWithLength:K * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_Y = [device newBufferWithLength:N * 2 options:MTLResourceStorageModeShared];
        fill_buffer_deterministic([buf_X contents], K * 2, 333);
        fill_buffer_deterministic([buf_Y contents], N * 2, 444);

        MTLSize gridSize = MTLSizeMake((N + 3) / 4, 1, 1);
        MTLSize tgSize   = MTLSizeMake(128, 1, 1);
        id<MTLComputePipelineState> pso = pso_int4_1row_tpl[group_size];

        std::vector<uint32_t> batch_sizes = {1, 2, 4, 8, 16, 32, 64};

        std::ofstream csv(csv_path);
        csv << "batch_size,gpu_median_us,host_median_us,host_overhead_ratio\n";

        for (uint32_t B : batch_sizes) {
            std::vector<double> gpu_latencies;
            std::vector<double> host_latencies;
            uint32_t total_dispatches = 128;
            uint32_t batches = total_dispatches / B;

            for (uint32_t b = 0; b < batches; ++b) {
                id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pso];

                for (uint32_t d = 0; d < B; ++d) {
                    uint32_t buf_idx = (b * B + d) % ring_size;
                    [enc setBuffer:weight_ring[buf_idx] offset:0 atIndex:0];
                    [enc setBuffer:scale_ring[buf_idx] offset:0 atIndex:1];
                    [enc setBuffer:buf_X offset:0 atIndex:2];
                    [enc setBuffer:buf_Y offset:0 atIndex:3];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
                    [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
                }
                [enc endEncoding];

                uint64_t t0 = mach_absolute_time();
                [cmd commit];
                [cmd waitUntilCompleted];
                uint64_t t1 = mach_absolute_time();

                double host_batch_us = ticksToUs(t1 - t0);
                double gpu_batch_us = ([cmd GPUEndTime] - [cmd GPUStartTime]) * 1e6;

                host_latencies.push_back(host_batch_us / (double)B);
                gpu_latencies.push_back(gpu_batch_us / (double)B);
            }

            std::sort(gpu_latencies.begin(), gpu_latencies.end());
            std::sort(host_latencies.begin(), host_latencies.end());
            double med_gpu = gpu_latencies[gpu_latencies.size() / 2];
            double med_host = host_latencies[host_latencies.size() / 2];
            double overhead = med_host / med_gpu;

            std::cout << "BatchSize: " << std::setw(2) << B
                      << " -> GPU: " << std::setw(6) << std::fixed << std::setprecision(2) << med_gpu << " us | "
                      << "Host CPU: " << std::setw(7) << med_host << " us | "
                      << "Overhead: " << std::setw(4) << std::setprecision(2) << overhead << "x" << std::endl;

            csv << B << "," << med_gpu << "," << med_host << "," << overhead << "\n";
        }
    }

    // Ablation 3: Fine-Grained Group Size Spectrum across 4 Kernel Architectures!
    void runFineGrainedGroupAblation(const std::string& csv_path) {
        std::cout << "\n==========================================================================================" << std::endl;
        std::cout << " [ABLATION 3] Fine-Grained Group Size Spectrum Across 4 Kernel Architectures (4608x4608)" << std::endl;
        std::cout << " Comparing: (A) Group-Strided Legacy | (B) Coalesced 1-Row DynDiv | (C) Coalesced 1-Row Tpl<G> | (D) Coalesced 4-Row Tpl<G>" << std::endl;
        std::cout << "==========================================================================================" << std::endl;

        const uint32_t K = 4608, N = 4608, ring_size = 32;
        std::vector<uint32_t> fine_groups = {16, 24, 32, 48, 64, 72, 96, 128, 256};

        std::ofstream csv(csv_path);
        csv << "group_size,scale_overhead_pct,"
            << "legacy_dram_us,legacy_dram_gflops,legacy_slc_us,legacy_slc_gflops,"
            << "coal1_dyn_dram_us,coal1_dyn_dram_gflops,coal1_dyn_slc_us,coal1_dyn_slc_gflops,"
            << "coal1_tpl_dram_us,coal1_tpl_dram_gflops,coal1_tpl_slc_us,coal1_tpl_slc_gflops,"
            << "coal4_tpl_dram_us,coal4_tpl_dram_gflops,coal4_tpl_slc_us,coal4_tpl_slc_gflops\n";

        for (uint32_t gs : fine_groups) {
            uint32_t num_groups = K / gs;
            double scale_overhead = ((double)(N * num_groups * 2) / (double)(N * (K / 2))) * 100.0;

            std::vector<id<MTLBuffer>> weight_ring(ring_size);
            std::vector<id<MTLBuffer>> scale_ring(ring_size);
            for (uint32_t b = 0; b < ring_size; ++b) {
                weight_ring[b] = [device newBufferWithLength:N * (K / 2) options:MTLResourceStorageModeShared];
                scale_ring[b]  = [device newBufferWithLength:N * num_groups * 2 options:MTLResourceStorageModeShared];
                fill_buffer_deterministic([weight_ring[b] contents], N * (K / 2), b + 111);
                fill_buffer_deterministic([scale_ring[b] contents], N * num_groups * 2, b + 222);
            }

            id<MTLBuffer> buf_X = [device newBufferWithLength:K * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_Y = [device newBufferWithLength:N * 2 options:MTLResourceStorageModeShared];
            fill_buffer_deterministic([buf_X contents], K * 2, 333);
            fill_buffer_deterministic([buf_Y contents], N * 2, 444);

            double flops = 2.0 * (double)N * (double)K;

            // (A) Legacy Group-Strided (1 row per 128-thread TG)
            double leg_dram_us = measureKernelLatencyUs(
                pso_int4_legacy, MTLSizeMake(N, 1, 1), MTLSizeMake(128, 1, 1),
                weight_ring, scale_ring, buf_X, buf_Y, K, N, gs, true, true);
            double leg_slc_us = measureKernelLatencyUs(
                pso_int4_legacy, MTLSizeMake(N, 1, 1), MTLSizeMake(128, 1, 1),
                weight_ring, scale_ring, buf_X, buf_Y, K, N, gs, true, false);

            // (B) Coalesced 1-Row Dynamic Division (4 rows per 128-thread TG)
            double c1d_dram_us = measureKernelLatencyUs(
                pso_int4_dyn, MTLSizeMake((N + 3) / 4, 1, 1), MTLSizeMake(128, 1, 1),
                weight_ring, scale_ring, buf_X, buf_Y, K, N, gs, true, true);
            double c1d_slc_us = measureKernelLatencyUs(
                pso_int4_dyn, MTLSizeMake((N + 3) / 4, 1, 1), MTLSizeMake(128, 1, 1),
                weight_ring, scale_ring, buf_X, buf_Y, K, N, gs, true, false);

            // (C) Coalesced 1-Row Compile-Time Template<G> (4 rows per 128-thread TG)
            double c1t_dram_us = measureKernelLatencyUs(
                pso_int4_1row_tpl[gs], MTLSizeMake((N + 3) / 4, 1, 1), MTLSizeMake(128, 1, 1),
                weight_ring, scale_ring, buf_X, buf_Y, K, N, gs, false, true);
            double c1t_slc_us = measureKernelLatencyUs(
                pso_int4_1row_tpl[gs], MTLSizeMake((N + 3) / 4, 1, 1), MTLSizeMake(128, 1, 1),
                weight_ring, scale_ring, buf_X, buf_Y, K, N, gs, false, false);

            // (D) Coalesced 4-Row Register-Tiled Template<G> (8 rows per 64-thread TG)
            double c4t_dram_us = measureKernelLatencyUs(
                pso_int4_4row_tpl[gs], MTLSizeMake((N + 7) / 8, 1, 1), MTLSizeMake(64, 1, 1),
                weight_ring, scale_ring, buf_X, buf_Y, K, N, gs, false, true);
            double c4t_slc_us = measureKernelLatencyUs(
                pso_int4_4row_tpl[gs], MTLSizeMake((N + 7) / 8, 1, 1), MTLSizeMake(64, 1, 1),
                weight_ring, scale_ring, buf_X, buf_Y, K, N, gs, false, false);

            auto to_gf = [&](double us) { return (flops / (us * 1e-6)) / 1e9; };

            std::cout << "G=" << std::setw(3) << gs << " (" << std::setw(4) << std::fixed << std::setprecision(1) << scale_overhead << "%) | "
                      << "DRAM: Leg=" << std::setprecision(2) << std::setw(5) << leg_dram_us << "us (" << std::setw(5) << (int)to_gf(leg_dram_us) << "GF) "
                      << "C1Dyn=" << std::setw(5) << c1d_dram_us << "us (" << std::setw(5) << (int)to_gf(c1d_dram_us) << "GF) "
                      << "C1Tpl=" << std::setw(5) << c1t_dram_us << "us (" << std::setw(5) << (int)to_gf(c1t_dram_us) << "GF) "
                      << "C4Tpl=" << std::setw(5) << c4t_dram_us << "us (" << std::setw(5) << (int)to_gf(c4t_dram_us) << "GF) | "
                      << "SLC: C1Tpl=" << std::setw(5) << c1t_slc_us << "us (" << std::setw(5) << (int)to_gf(c1t_slc_us) << "GF) "
                      << "C4Tpl=" << std::setw(5) << c4t_slc_us << "us (" << std::setw(5) << (int)to_gf(c4t_slc_us) << "GF)"
                      << std::endl;

            csv << gs << "," << scale_overhead << ","
                << leg_dram_us << "," << to_gf(leg_dram_us) << "," << leg_slc_us << "," << to_gf(leg_slc_us) << ","
                << c1d_dram_us << "," << to_gf(c1d_dram_us) << "," << c1d_slc_us << "," << to_gf(c1d_slc_us) << ","
                << c1t_dram_us << "," << to_gf(c1t_dram_us) << "," << c1t_slc_us << "," << to_gf(c1t_slc_us) << ","
                << c4t_dram_us << "," << to_gf(c4t_dram_us) << "," << c4t_slc_us << "," << to_gf(c4t_slc_us) << "\n";
        }
    }
};

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        try {
            AblationSuite suite;
            std::string results_dir = "/Users/fo/Documents/Antigravity Test/metal_unified_memory_quant_study/results";

            suite.runRingSizeAblation(results_dir + "/ablation_ring_pool_slc.csv");
            suite.runBatchSizeAblation(results_dir + "/ablation_batch_dispatch.csv");
            suite.runFineGrainedGroupAblation(results_dir + "/ablation_group_granularity.csv");

            std::cout << "\n[+] All Ablation Studies Completed Successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Ablation Error: " << e.what() << std::endl;
            return 1;
        }
    }
    return 0;
}
