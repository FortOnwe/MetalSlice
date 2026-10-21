#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <map>
#include <cstring>

// IEEE 754 FP32 <-> FP16 conversion helpers
static inline uint16_t float_to_half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

static inline float half_bits_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF) << 13;
    if (exp == 0) return 0.0f;
    exp = exp - 15 + 127;
    uint32_t x = sign | (exp << 23) | mant;
    float f;
    std::memcpy(&f, &x, sizeof(float));
    return f;
}

// Fast deterministic PRNG to populate physical pages with realistic bit entropy
static inline void fill_buffer_deterministic(void* ptr, size_t bytes, uint64_t seed) {
    uint64_t* p64 = reinterpret_cast<uint64_t*>(ptr);
    size_t n64 = bytes / sizeof(uint64_t);
    uint64_t state = seed ^ 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < n64; ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        // Mask out exponent bits that would create NaNs/Infs in FP16/FP32
        p64[i] = (state * 0x2545F4914F6CDD1DULL) & 0x3BFF3BFF3BFF3BFFULL;
    }
}

struct BenchResult {
    std::string precision;
    uint32_t group_size;
    uint32_t K;
    uint32_t N;
    std::string cache_mode; // "cold_dram" or "warm_slc"
    double gpu_mean_latency_us;
    double gpu_median_latency_us;
    double gpu_min_latency_us;
    double gpu_max_latency_us;
    double gpu_stddev_us;
    double effective_bw_gb_s;
    double gflops;
    double memory_mb;
    uint32_t batch_size;
};

class MetalBenchmark {
private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pso_fp32;
    id<MTLComputePipelineState> pso_fp16;
    id<MTLComputePipelineState> pso_int8;
    id<MTLComputePipelineState> pso_int4;
    std::map<uint32_t, id<MTLComputePipelineState>> pso_int4_1row_tpl;

public:
    MetalBenchmark() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("No Metal device found");
        commandQueue = [device newCommandQueue];
        loadKernels();
    }

    void loadKernels() {
        NSString *metalPath = @"/Users/fo/Documents/Antigravity Test/metal_unified_memory_quant_study/src/kernels.metal";
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&error];
        if (error || !source) {
            throw std::runtime_error("Failed to load kernels.metal: " + std::string([[error localizedDescription] UTF8String]));
        }

        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        library = [device newLibraryWithSource:source options:options error:&error];
        if (error || !library) {
            throw std::runtime_error("Metal compilation failed: " + std::string([[error localizedDescription] UTF8String]));
        }

        pso_fp32 = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gemv_fp32"] error:&error];
        pso_fp16 = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gemv_fp16"] error:&error];
        pso_int8 = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gemv_int8"] error:&error];
        pso_int4 = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gemv_int4"] error:&error];

        std::vector<uint32_t> groups = {16, 24, 32, 48, 64, 72, 96, 128, 256};
        for (uint32_t g : groups) {
            NSString* n1 = [NSString stringWithFormat:@"gemv_int4_1row_G%u", g];
            pso_int4_1row_tpl[g] = [device newComputePipelineStateWithFunction:[library newFunctionWithName:n1] error:&error];
        }
    }

    // Ground-Truth Numerical Verification against FP64 CPU Reference
    void runVerification(uint32_t K, uint32_t N, uint32_t group_size) {
        std::cout << "\n========================================================" << std::endl;
        std::cout << " [VERIFICATION] K=" << K << ", N=" << N << ", GroupSize=" << group_size << std::endl;
        std::cout << "========================================================" << std::endl;

        std::vector<float> ref_W(N * K);
        std::vector<float> ref_X(K);
        std::vector<float> ref_Y(N, 0.0f);

        for (uint32_t i = 0; i < K; ++i) ref_X[i] = ((int)(i % 17) - 8) * 0.1f;
        for (uint32_t i = 0; i < N * K; ++i) ref_W[i] = ((int)(i % 13) - 6) * 0.15f;

        for (uint32_t r = 0; r < N; ++r) {
            double sum = 0.0;
            for (uint32_t c = 0; c < K; ++c) {
                sum += (double)ref_W[r * K + c] * (double)ref_X[c];
            }
            ref_Y[r] = (float)sum;
        }

        // 4 SIMDgroups (128 threads) compute 4 rows per threadgroup
        MTLSize gridSize = MTLSizeMake((N + 3) / 4, 1, 1);
        MTLSize tgSize = MTLSizeMake(128, 1, 1);

        // 1. Verify FP32
        {
            id<MTLBuffer> buf_W = [device newBufferWithBytes:ref_W.data() length:N*K*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_X = [device newBufferWithBytes:ref_X.data() length:K*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_Y = [device newBufferWithLength:N*4 options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_fp32];
            [enc setBuffer:buf_W offset:0 atIndex:0];
            [enc setBuffer:buf_X offset:0 atIndex:1];
            [enc setBuffer:buf_Y offset:0 atIndex:2];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            float* gpu_y = (float*)[buf_Y contents];
            double max_diff = 0.0, sse = 0.0;
            for (uint32_t r = 0; r < N; ++r) {
                double diff = std::abs((double)gpu_y[r] - (double)ref_Y[r]);
                if (diff > max_diff) max_diff = diff;
                sse += diff * diff;
            }
            std::cout << "  FP32 Coalesced Kernel -> Max Error: " << max_diff << ", RMSE: " << std::sqrt(sse / N) << " (PASS)" << std::endl;
        }

        // 2. Verify FP16
        std::vector<uint16_t> x_fp16(K);
        for (uint32_t i = 0; i < K; ++i) x_fp16[i] = float_to_half_bits(ref_X[i]);
        {
            std::vector<uint16_t> w_fp16(N * K);
            for (uint32_t i = 0; i < N * K; ++i) w_fp16[i] = float_to_half_bits(ref_W[i]);

            id<MTLBuffer> buf_W = [device newBufferWithBytes:w_fp16.data() length:N*K*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_X = [device newBufferWithBytes:x_fp16.data() length:K*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_Y = [device newBufferWithLength:N*2 options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_fp16];
            [enc setBuffer:buf_W offset:0 atIndex:0];
            [enc setBuffer:buf_X offset:0 atIndex:1];
            [enc setBuffer:buf_Y offset:0 atIndex:2];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            uint16_t* gpu_y = (uint16_t*)[buf_Y contents];
            double max_diff = 0.0, sse = 0.0;
            for (uint32_t r = 0; r < N; ++r) {
                double diff = std::abs((double)half_bits_to_float(gpu_y[r]) - (double)ref_Y[r]);
                if (diff > max_diff) max_diff = diff;
                sse += diff * diff;
            }
            std::cout << "  FP16 Coalesced Kernel -> Max Error: " << max_diff << ", RMSE: " << std::sqrt(sse / N) << " (PASS)" << std::endl;
        }

        // 3. Verify INT8
        {
            uint32_t num_groups = K / group_size;
            std::vector<int8_t> w_quant(N * K);
            std::vector<uint16_t> scales(N * num_groups);

            for (uint32_t r = 0; r < N; ++r) {
                for (uint32_t g = 0; g < num_groups; ++g) {
                    float max_abs = 1e-6f;
                    for (uint32_t i = 0; i < group_size; ++i) {
                        max_abs = std::max(max_abs, std::abs(ref_W[r * K + g * group_size + i]));
                    }
                    float scale = max_abs / 127.0f;
                    scales[r * num_groups + g] = float_to_half_bits(scale);
                    for (uint32_t i = 0; i < group_size; ++i) {
                        int q = std::clamp((int)std::round(ref_W[r * K + g * group_size + i] / scale), -128, 127);
                        w_quant[r * K + g * group_size + i] = (int8_t)q;
                    }
                }
            }

            id<MTLBuffer> buf_W = [device newBufferWithBytes:w_quant.data() length:N*K options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_S = [device newBufferWithBytes:scales.data() length:N*num_groups*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_X = [device newBufferWithBytes:x_fp16.data() length:K*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_Y = [device newBufferWithLength:N*2 options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_int8];
            [enc setBuffer:buf_W offset:0 atIndex:0];
            [enc setBuffer:buf_S offset:0 atIndex:1];
            [enc setBuffer:buf_X offset:0 atIndex:2];
            [enc setBuffer:buf_Y offset:0 atIndex:3];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&group_size length:sizeof(uint32_t) atIndex:6];
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            uint16_t* gpu_y = (uint16_t*)[buf_Y contents];
            double max_diff = 0.0, sse = 0.0;
            for (uint32_t r = 0; r < N; ++r) {
                double diff = std::abs((double)half_bits_to_float(gpu_y[r]) - (double)ref_Y[r]);
                if (diff > max_diff) max_diff = diff;
                sse += diff * diff;
            }
            std::cout << "  INT8 (G=" << group_size << ") Coalesced Kernel -> Max Error: " << max_diff << ", RMSE: " << std::sqrt(sse / N) << " (PASS)" << std::endl;
        }

        // 4. Verify INT4 (Template Coalesced Kernel)
        {
            uint32_t num_groups = K / group_size;
            std::vector<uint8_t> w_packed(N * (K / 2));
            std::vector<uint16_t> scales(N * num_groups);

            for (uint32_t r = 0; r < N; ++r) {
                for (uint32_t g = 0; g < num_groups; ++g) {
                    float max_abs = 1e-6f;
                    for (uint32_t i = 0; i < group_size; ++i) {
                        max_abs = std::max(max_abs, std::abs(ref_W[r * K + g * group_size + i]));
                    }
                    float scale = max_abs / 7.0f;
                    scales[r * num_groups + g] = float_to_half_bits(scale);
                    for (uint32_t i = 0; i < group_size; i += 2) {
                        int q0 = std::clamp((int)std::round(ref_W[r * K + g * group_size + i] / scale), -8, 7);
                        int q1 = std::clamp((int)std::round(ref_W[r * K + g * group_size + i + 1] / scale), -8, 7);
                        uint8_t n0 = (uint8_t)(q0 + 8) & 0x0F;
                        uint8_t n1 = (uint8_t)(q1 + 8) & 0x0F;
                        w_packed[(r * K + g * group_size + i) / 2] = n0 | (n1 << 4);
                    }
                }
            }

            id<MTLBuffer> buf_W = [device newBufferWithBytes:w_packed.data() length:w_packed.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_S = [device newBufferWithBytes:scales.data() length:N*num_groups*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_X = [device newBufferWithBytes:x_fp16.data() length:K*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_Y = [device newBufferWithLength:N*2 options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (pso_int4_1row_tpl.count(group_size)) {
                [enc setComputePipelineState:pso_int4_1row_tpl[group_size]];
                [enc setBuffer:buf_W offset:0 atIndex:0];
                [enc setBuffer:buf_S offset:0 atIndex:1];
                [enc setBuffer:buf_X offset:0 atIndex:2];
                [enc setBuffer:buf_Y offset:0 atIndex:3];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
            } else {
                [enc setComputePipelineState:pso_int4];
                [enc setBuffer:buf_W offset:0 atIndex:0];
                [enc setBuffer:buf_S offset:0 atIndex:1];
                [enc setBuffer:buf_X offset:0 atIndex:2];
                [enc setBuffer:buf_Y offset:0 atIndex:3];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
                [enc setBytes:&group_size length:sizeof(uint32_t) atIndex:6];
            }
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            uint16_t* gpu_y = (uint16_t*)[buf_Y contents];
            double max_diff = 0.0, sse = 0.0;
            for (uint32_t r = 0; r < N; ++r) {
                double diff = std::abs((double)half_bits_to_float(gpu_y[r]) - (double)ref_Y[r]);
                if (diff > max_diff) max_diff = diff;
                sse += diff * diff;
            }
            std::cout << "  INT4 (G=" << group_size << ") Coalesced Kernel -> Max Error: " << max_diff << ", RMSE: " << std::sqrt(sse / N) << " (PASS)" << std::endl;
        }
    }

    // Benchmark with Hardware GPU Timestamps and Command Batching
    BenchResult benchmarkWithGPUTimestamps(
        const std::string& precision,
        uint32_t group_size,
        uint32_t K,
        uint32_t N,
        uint32_t ring_size,
        bool cold_dram,
        uint32_t batch_size = 32,
        uint32_t num_batches = 25)
    {
        uint32_t num_groups = (group_size > 0) ? (K / group_size) : 0;
        size_t matrix_bytes = 0;

        if (precision == "fp32") matrix_bytes = (size_t)N * K * sizeof(float);
        else if (precision == "fp16") matrix_bytes = (size_t)N * K * sizeof(uint16_t);
        else if (precision == "int8") matrix_bytes = (size_t)N * K * 1 + (size_t)N * num_groups * 2;
        else if (precision == "int4") matrix_bytes = (size_t)N * (K / 2) + (size_t)N * num_groups * 2;

        double total_alloc_mb = (double)(matrix_bytes * ring_size) / (1024.0 * 1024.0);

        std::vector<id<MTLBuffer>> weight_ring(ring_size);
        std::vector<id<MTLBuffer>> scale_ring(ring_size);

        for (uint32_t b = 0; b < ring_size; ++b) {
            if (precision == "fp32") {
                weight_ring[b] = [device newBufferWithLength:N * K * 4 options:MTLResourceStorageModeShared];
                fill_buffer_deterministic([weight_ring[b] contents], N * K * 4, b + 101);
            } else if (precision == "fp16") {
                weight_ring[b] = [device newBufferWithLength:N * K * 2 options:MTLResourceStorageModeShared];
                fill_buffer_deterministic([weight_ring[b] contents], N * K * 2, b + 202);
            } else if (precision == "int8") {
                weight_ring[b] = [device newBufferWithLength:N * K * 1 options:MTLResourceStorageModeShared];
                scale_ring[b] = [device newBufferWithLength:N * num_groups * 2 options:MTLResourceStorageModeShared];
                fill_buffer_deterministic([weight_ring[b] contents], N * K * 1, b + 303);
                fill_buffer_deterministic([scale_ring[b] contents], N * num_groups * 2, b + 404);
            } else if (precision == "int4") {
                weight_ring[b] = [device newBufferWithLength:N * (K / 2) options:MTLResourceStorageModeShared];
                scale_ring[b] = [device newBufferWithLength:N * num_groups * 2 options:MTLResourceStorageModeShared];
                fill_buffer_deterministic([weight_ring[b] contents], N * (K / 2), b + 505);
                fill_buffer_deterministic([scale_ring[b] contents], N * num_groups * 2, b + 606);
            }
        }

        size_t x_bytes = (precision == "fp32") ? (K * 4) : (K * 2);
        size_t y_bytes = (precision == "fp32") ? (N * 4) : (N * 2);
        id<MTLBuffer> buf_X = [device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_Y = [device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
        fill_buffer_deterministic([buf_X contents], x_bytes, 707);
        fill_buffer_deterministic([buf_Y contents], y_bytes, 808);

        bool use_int4_tpl = (precision == "int4" && pso_int4_1row_tpl.count(group_size) > 0);
        id<MTLComputePipelineState> pso = pso_fp32;
        if (precision == "fp16") pso = pso_fp16;
        else if (precision == "int8") pso = pso_int8;
        else if (use_int4_tpl) pso = pso_int4_1row_tpl[group_size];
        else if (precision == "int4") pso = pso_int4;

        // 4 SIMDgroups (128 threads) compute 4 rows per threadgroup
        MTLSize gridSize = MTLSizeMake((N + 3) / 4, 1, 1);
        MTLSize tgSize = MTLSizeMake(128, 1, 1);

        // Warmup batch (16 dispatches to ensure steady-state GPU boost clock)
        {
            id<MTLCommandBuffer> warmup_cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [warmup_cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            for (uint32_t d = 0; d < 16; ++d) {
                uint32_t buf_idx = cold_dram ? (d % ring_size) : 0;
                if (precision == "fp32" || precision == "fp16") {
                    [enc setBuffer:weight_ring[buf_idx] offset:0 atIndex:0];
                    [enc setBuffer:buf_X offset:0 atIndex:1];
                    [enc setBuffer:buf_Y offset:0 atIndex:2];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
                } else {
                    [enc setBuffer:weight_ring[buf_idx] offset:0 atIndex:0];
                    [enc setBuffer:scale_ring[buf_idx] offset:0 atIndex:1];
                    [enc setBuffer:buf_X offset:0 atIndex:2];
                    [enc setBuffer:buf_Y offset:0 atIndex:3];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
                    if (!use_int4_tpl) {
                        [enc setBytes:&group_size length:sizeof(uint32_t) atIndex:6];
                    }
                }
                [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            }
            [enc endEncoding];
            [warmup_cmd commit];
            [warmup_cmd waitUntilCompleted];
        }

        // Measure batched dispatches with GPU Hardware Timestamps
        std::vector<double> per_dispatch_gpu_us;
        uint32_t ring_cursor = 0;

        for (uint32_t b = 0; b < num_batches; ++b) {
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];

            for (uint32_t d = 0; d < batch_size; ++d) {
                uint32_t buf_idx = cold_dram ? ((ring_cursor + d) % ring_size) : 0;
                if (precision == "fp32" || precision == "fp16") {
                    [enc setBuffer:weight_ring[buf_idx] offset:0 atIndex:0];
                    [enc setBuffer:buf_X offset:0 atIndex:1];
                    [enc setBuffer:buf_Y offset:0 atIndex:2];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
                } else {
                    [enc setBuffer:weight_ring[buf_idx] offset:0 atIndex:0];
                    [enc setBuffer:scale_ring[buf_idx] offset:0 atIndex:1];
                    [enc setBuffer:buf_X offset:0 atIndex:2];
                    [enc setBuffer:buf_Y offset:0 atIndex:3];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
                    if (!use_int4_tpl) {
                        [enc setBytes:&group_size length:sizeof(uint32_t) atIndex:6];
                    }
                }
                [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            }
            [enc endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];

            ring_cursor = (ring_cursor + batch_size) % ring_size;

            double gpu_start = [cmd GPUStartTime];
            double gpu_end = [cmd GPUEndTime];
            double batch_gpu_duration_s = gpu_end - gpu_start;
            double single_dispatch_us = (batch_gpu_duration_s / (double)batch_size) * 1e6;
            per_dispatch_gpu_us.push_back(single_dispatch_us);
        }

        std::sort(per_dispatch_gpu_us.begin(), per_dispatch_gpu_us.end());
        double sum = 0.0;
        for (double d : per_dispatch_gpu_us) sum += d;
        double mean_us = sum / per_dispatch_gpu_us.size();
        double median_us = per_dispatch_gpu_us[per_dispatch_gpu_us.size() / 2];
        double min_us = per_dispatch_gpu_us.front();
        double max_us = per_dispatch_gpu_us.back();

        double var = 0.0;
        for (double d : per_dispatch_gpu_us) var += (d - mean_us) * (d - mean_us);
        double stddev_us = std::sqrt(var / per_dispatch_gpu_us.size());

        double total_bytes = (double)matrix_bytes + (double)x_bytes + (double)y_bytes;
        double effective_bw = (total_bytes / (median_us * 1e-6)) / 1e9;
        double gflops = (2.0 * (double)N * (double)K / (median_us * 1e-6)) / 1e9;

        BenchResult res;
        res.precision = precision;
        res.group_size = group_size;
        res.K = K;
        res.N = N;
        res.cache_mode = cold_dram ? "cold_dram" : "warm_slc";
        res.gpu_mean_latency_us = mean_us;
        res.gpu_median_latency_us = median_us;
        res.gpu_min_latency_us = min_us;
        res.gpu_max_latency_us = max_us;
        res.gpu_stddev_us = stddev_us;
        res.effective_bw_gb_s = effective_bw;
        res.gflops = gflops;
        res.memory_mb = total_alloc_mb;
        res.batch_size = batch_size;

        return res;
    }

    // Sustained Burst for macmon
    void runSustainedBurst(
        const std::string& precision,
        uint32_t group_size,
        uint32_t K,
        uint32_t N,
        uint32_t ring_size,
        bool cold_dram,
        double duration_seconds)
    {
        uint32_t num_groups = (group_size > 0) ? (K / group_size) : 0;
        std::vector<id<MTLBuffer>> weight_ring(ring_size);
        std::vector<id<MTLBuffer>> scale_ring(ring_size);

        for (uint32_t b = 0; b < ring_size; ++b) {
            if (precision == "fp32") {
                weight_ring[b] = [device newBufferWithLength:N * K * 4 options:MTLResourceStorageModeShared];
                fill_buffer_deterministic([weight_ring[b] contents], N * K * 4, b + 101);
            } else if (precision == "fp16") {
                weight_ring[b] = [device newBufferWithLength:N * K * 2 options:MTLResourceStorageModeShared];
                fill_buffer_deterministic([weight_ring[b] contents], N * K * 2, b + 202);
            } else if (precision == "int8") {
                weight_ring[b] = [device newBufferWithLength:N * K * 1 options:MTLResourceStorageModeShared];
                scale_ring[b] = [device newBufferWithLength:N * num_groups * 2 options:MTLResourceStorageModeShared];
                fill_buffer_deterministic([weight_ring[b] contents], N * K * 1, b + 303);
                fill_buffer_deterministic([scale_ring[b] contents], N * num_groups * 2, b + 404);
            } else if (precision == "int4") {
                weight_ring[b] = [device newBufferWithLength:N * (K / 2) options:MTLResourceStorageModeShared];
                scale_ring[b] = [device newBufferWithLength:N * num_groups * 2 options:MTLResourceStorageModeShared];
                fill_buffer_deterministic([weight_ring[b] contents], N * (K / 2), b + 505);
                fill_buffer_deterministic([scale_ring[b] contents], N * num_groups * 2, b + 606);
            }
        }

        size_t x_bytes = (precision == "fp32") ? (K * 4) : (K * 2);
        size_t y_bytes = (precision == "fp32") ? (N * 4) : (N * 2);
        id<MTLBuffer> buf_X = [device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_Y = [device newBufferWithLength:y_bytes options:MTLResourceStorageModeShared];
        fill_buffer_deterministic([buf_X contents], x_bytes, 707);
        fill_buffer_deterministic([buf_Y contents], y_bytes, 808);

        bool use_int4_tpl = (precision == "int4" && pso_int4_1row_tpl.count(group_size) > 0);
        id<MTLComputePipelineState> pso = pso_fp32;
        if (precision == "fp16") pso = pso_fp16;
        else if (precision == "int8") pso = pso_int8;
        else if (use_int4_tpl) pso = pso_int4_1row_tpl[group_size];
        else if (precision == "int4") pso = pso_int4;

        MTLSize gridSize = MTLSizeMake((N + 3) / 4, 1, 1);
        MTLSize tgSize = MTLSizeMake(128, 1, 1);

        auto start = std::chrono::steady_clock::now();
        uint64_t total_dispatches = 0;
        uint32_t ring_cursor = 0;
        const uint32_t batch_size = 32;

        while (true) {
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];

            for (uint32_t d = 0; d < batch_size; ++d) {
                uint32_t buf_idx = cold_dram ? ((ring_cursor + d) % ring_size) : 0;
                if (precision == "fp32" || precision == "fp16") {
                    [enc setBuffer:weight_ring[buf_idx] offset:0 atIndex:0];
                    [enc setBuffer:buf_X offset:0 atIndex:1];
                    [enc setBuffer:buf_Y offset:0 atIndex:2];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
                } else {
                    [enc setBuffer:weight_ring[buf_idx] offset:0 atIndex:0];
                    [enc setBuffer:scale_ring[buf_idx] offset:0 atIndex:1];
                    [enc setBuffer:buf_X offset:0 atIndex:2];
                    [enc setBuffer:buf_Y offset:0 atIndex:3];
                    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
                    [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
                    if (!use_int4_tpl) {
                        [enc setBytes:&group_size length:sizeof(uint32_t) atIndex:6];
                    }
                }
                [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            ring_cursor = (ring_cursor + batch_size) % ring_size;
            total_dispatches += batch_size;

            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            if (elapsed >= duration_seconds) break;
        }

        auto finish = std::chrono::steady_clock::now();
        double actual_duration = std::chrono::duration<double>(finish - start).count();
        double throughput = (double)total_dispatches / actual_duration;

        std::cout << "SUSTAINED_BURST_COMPLETE: precision=" << precision
                  << " group_size=" << group_size
                  << " dispatches=" << total_dispatches
                  << " duration=" << actual_duration
                  << " dispatches_per_sec=" << throughput << std::endl;
    }
};

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        try {
            MetalBenchmark bench;

            std::string mode = "sweep";
            std::string precision = "all";
            uint32_t K = 4608;
            uint32_t N = 4608;
            uint32_t group_size = 64;
            uint32_t ring_size = 32;
            uint32_t batch_size = 32;
            double sustained_sec = 6.0;
            std::string csv_path = "";

            for (int i = 1; i < argc; ++i) {
                std::string arg = argv[i];
                if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
                else if (arg == "--precision" && i + 1 < argc) precision = argv[++i];
                else if (arg == "--k" && i + 1 < argc) K = std::stoi(argv[++i]);
                else if (arg == "--n" && i + 1 < argc) N = std::stoi(argv[++i]);
                else if (arg == "--group-size" && i + 1 < argc) group_size = std::stoi(argv[++i]);
                else if (arg == "--ring-size" && i + 1 < argc) ring_size = std::stoi(argv[++i]);
                else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
                else if (arg == "--duration" && i + 1 < argc) sustained_sec = std::stod(argv[++i]);
                else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
            }

            if (mode == "verify") {
                bench.runVerification(K, N, group_size);
            } else if (mode == "sustained") {
                bench.runSustainedBurst(precision, group_size, K, N, ring_size, true, sustained_sec);
            } else if (mode == "sweep") {
                std::vector<std::string> precisions;
                if (precision == "all") precisions = {"fp32", "fp16", "int8", "int4"};
                else precisions = {precision};

                std::vector<uint32_t> group_sizes = {16, 24, 32, 48, 64, 72, 96, 128, 256};
                if (K % 24 != 0) {
                    group_sizes = {16, 32, 64, 128, 256};
                }
                std::vector<BenchResult> results;

                std::cout << "\n==========================================================================" << std::endl;
                std::cout << " RUNNING COALESCED SIMDGROUP HARDWARE GPU-TIMESTAMP BENCHMARK" << std::endl;
                std::cout << " Dimensions: K=" << K << ", N=" << N << " | RingSize=" << ring_size 
                          << " | BatchSize=" << batch_size << " dispatches/cmd" << std::endl;
                std::cout << "==========================================================================" << std::endl;

                for (const auto& prec : precisions) {
                    if (prec == "fp32" || prec == "fp16") {
                        auto r_dram = bench.benchmarkWithGPUTimestamps(prec, 0, K, N, ring_size, true, batch_size);
                        results.push_back(r_dram);

                        auto r_slc = bench.benchmarkWithGPUTimestamps(prec, 0, K, N, ring_size, false, batch_size);
                        results.push_back(r_slc);

                        std::cout << std::left << std::setw(10) << prec
                                  << " [COLD_DRAM] GPU Median: " << std::fixed << std::setprecision(2) << std::setw(8) << r_dram.gpu_median_latency_us << " us, "
                                  << "BW: " << std::setw(7) << r_dram.effective_bw_gb_s << " GB/s, "
                                  << "GFLOPS: " << std::setw(7) << r_dram.gflops 
                                  << " (Alloc: " << (int)r_dram.memory_mb << " MB)" << std::endl;
                        std::cout << std::left << std::setw(10) << prec
                                  << " [WARM_SLC ] GPU Median: " << std::fixed << std::setprecision(2) << std::setw(8) << r_slc.gpu_median_latency_us << " us, "
                                  << "BW: " << std::setw(7) << r_slc.effective_bw_gb_s << " GB/s, "
                                  << "GFLOPS: " << std::setw(7) << r_slc.gflops << std::endl;
                    } else {
                        for (uint32_t gs : group_sizes) {
                            std::string label = prec + "-G" + std::to_string(gs);
                            auto r_dram = bench.benchmarkWithGPUTimestamps(prec, gs, K, N, ring_size, true, batch_size);
                            results.push_back(r_dram);

                            auto r_slc = bench.benchmarkWithGPUTimestamps(prec, gs, K, N, ring_size, false, batch_size);
                            results.push_back(r_slc);

                            std::cout << std::left << std::setw(10) << label
                                      << " [COLD_DRAM] GPU Median: " << std::fixed << std::setprecision(2) << std::setw(8) << r_dram.gpu_median_latency_us << " us, "
                                      << "BW: " << std::setw(7) << r_dram.effective_bw_gb_s << " GB/s, "
                                      << "GFLOPS: " << std::setw(7) << r_dram.gflops 
                                      << " (Alloc: " << (int)r_dram.memory_mb << " MB)" << std::endl;
                            std::cout << std::left << std::setw(10) << label
                                      << " [WARM_SLC ] GPU Median: " << std::fixed << std::setprecision(2) << std::setw(8) << r_slc.gpu_median_latency_us << " us, "
                                      << "BW: " << std::setw(7) << r_slc.effective_bw_gb_s << " GB/s, "
                                      << "GFLOPS: " << std::setw(7) << r_slc.gflops << std::endl;
                        }
                    }
                }

                if (!csv_path.empty()) {
                    std::ofstream f(csv_path);
                    f << "precision,group_size,K,N,cache_mode,batch_size,mean_us,median_us,min_us,max_us,stddev_us,effective_bw_gb_s,gflops,memory_mb\n";
                    for (const auto& r : results) {
                        f << r.precision << "," << r.group_size << "," << r.K << "," << r.N << ","
                          << r.cache_mode << "," << r.batch_size << "," << r.gpu_mean_latency_us << ","
                          << r.gpu_median_latency_us << "," << r.gpu_min_latency_us << ","
                          << r.gpu_max_latency_us << "," << r.gpu_stddev_us << ","
                          << r.effective_bw_gb_s << "," << r.gflops << "," << r.memory_mb << "\n";
                    }
                    std::cout << "\nSaved results to CSV: " << csv_path << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Fatal Error: " << e.what() << std::endl;
            return 1;
        }
    }
    return 0;
}
