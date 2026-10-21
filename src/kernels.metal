#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ============================================================================
// Constant Definitions
// ============================================================================
constant uint SIMD_SIZE = 32;
constant uint ROWS_PER_SIMD = 4;
constant uint SIMDS_PER_TG  = 2;
constant uint ROWS_PER_TG   = ROWS_PER_SIMD * SIMDS_PER_TG; // 8 rows per 64-thread threadgroup

// ============================================================================
// 1. FP32 Matrix-Vector Multiplication (GEMV) — Coalesced 1-Row-Per-SIMDgroup
// 4 SIMDgroups (128 threads) compute 4 rows per threadgroup; 100% coalesced float4 loads.
// ============================================================================
kernel void gemv_fp32(
    device const float*  W       [[buffer(0)]], // [N, K]
    device const float*  X       [[buffer(1)]], // [K]
    device float*        Y       [[buffer(2)]], // [N]
    constant uint&       K       [[buffer(3)]],
    constant uint&       N       [[buffer(4)]],
    uint2                gid     [[threadgroup_position_in_grid]],
    uint                 simd_id [[simdgroup_index_in_threadgroup]],
    uint                 lane_id [[thread_index_in_simdgroup]])
{
    uint row = gid.x * 4 + simd_id;
    if (row >= N) return;

    device const float4* W_row = reinterpret_cast<device const float4*>(W + row * K);
    device const float4* X_vec = reinterpret_cast<device const float4*>(X);
    uint num_vec4 = K / 4;

    float acc = 0.0f;
    for (uint i = lane_id; i < num_vec4; i += SIMD_SIZE) {
        acc += dot(W_row[i], X_vec[i]);
    }

    acc = simd_sum(acc);
    if (lane_id == 0) {
        Y[row] = acc;
    }
}

// ============================================================================
// 2. FP16 Matrix-Vector Multiplication (GEMV) — Coalesced 1-Row-Per-SIMDgroup
// 4 SIMDgroups (128 threads) compute 4 rows per threadgroup; 100% coalesced 2x half4 loads.
// ============================================================================
kernel void gemv_fp16(
    device const half*   W       [[buffer(0)]], // [N, K]
    device const half*   X       [[buffer(1)]], // [K]
    device half*         Y       [[buffer(2)]], // [N]
    constant uint&       K       [[buffer(3)]],
    constant uint&       N       [[buffer(4)]],
    uint2                gid     [[threadgroup_position_in_grid]],
    uint                 simd_id [[simdgroup_index_in_threadgroup]],
    uint                 lane_id [[thread_index_in_simdgroup]])
{
    uint row = gid.x * 4 + simd_id;
    if (row >= N) return;

    device const half4* W_row = reinterpret_cast<device const half4*>(W + row * K);
    device const half4* X_vec = reinterpret_cast<device const half4*>(X);
    uint num_vec4 = K / 4;

    float acc = 0.0f;
    for (uint i = lane_id * 2; i < num_vec4; i += SIMD_SIZE * 2) {
        acc += float(dot(W_row[i], X_vec[i])) + float(dot(W_row[i + 1], X_vec[i + 1]));
    }

    acc = simd_sum(acc);
    if (lane_id == 0) {
        Y[row] = static_cast<half>(acc);
    }
}

// ============================================================================
// 3. INT8 Block-Quantized GEMV — Coalesced 1-Row-Per-SIMDgroup
// ============================================================================
kernel void gemv_int8(
    device const char*   W_quant    [[buffer(0)]], // [N, K] int8 weights
    device const half*   Scales     [[buffer(1)]], // [N, K / group_size] scales
    device const half*   X          [[buffer(2)]], // [K]
    device half*         Y          [[buffer(3)]], // [N]
    constant uint&       K          [[buffer(4)]],
    constant uint&       N          [[buffer(5)]],
    constant uint&       group_size [[buffer(6)]],
    uint2                gid        [[threadgroup_position_in_grid]],
    uint                 simd_id    [[simdgroup_index_in_threadgroup]],
    uint                 lane_id    [[thread_index_in_simdgroup]])
{
    uint row = gid.x * 4 + simd_id;
    if (row >= N) return;

    device const char* row_w = W_quant + row * K;
    device const half* row_s = Scales + row * (K / group_size);

    float acc = 0.0f;

    for (uint elem_idx = lane_id * 8; elem_idx < K; elem_idx += SIMD_SIZE * 8) {
        uint g = elem_idx / group_size;
        float scale = float(row_s[g]);

        device const char4* w_vec = reinterpret_cast<device const char4*>(row_w + elem_idx);
        device const half4* x_vec = reinterpret_cast<device const half4*>(X + elem_idx);

        float sub = dot(float4(w_vec[0]), float4(x_vec[0])) +
                    dot(float4(w_vec[1]), float4(x_vec[1]));

        acc = fma(sub, scale, acc);
    }

    acc = simd_sum(acc);
    if (lane_id == 0) {
        Y[row] = static_cast<half>(acc);
    }
}

// ============================================================================
// 4. INT4 Block-Quantized GEMV — Coalesced 1-Row-Per-SIMDgroup (Dynamic G)
// ============================================================================
kernel void gemv_int4(
    device const uchar*  W_packed   [[buffer(0)]], // [N, K / 2] packed nibbles
    device const half*   Scales     [[buffer(1)]], // [N, K / group_size] scales
    device const half*   X          [[buffer(2)]], // [K]
    device half*         Y          [[buffer(3)]], // [N]
    constant uint&       K          [[buffer(4)]],
    constant uint&       N          [[buffer(5)]],
    constant uint&       group_size [[buffer(6)]],
    uint2                gid        [[threadgroup_position_in_grid]],
    uint                 simd_id    [[simdgroup_index_in_threadgroup]],
    uint                 lane_id    [[thread_index_in_simdgroup]])
{
    uint row = gid.x * 4 + simd_id;
    if (row >= N) return;

    device const uchar* row_w = W_packed + row * (K >> 1);
    device const half*  row_s = Scales + row * (K / group_size);

    float acc = 0.0f;

    for (uint elem_idx = lane_id * 8; elem_idx < K; elem_idx += SIMD_SIZE * 8) {
        uint g = elem_idx / group_size;
        float scale = float(row_s[g]);

        uchar4 packed = *reinterpret_cast<device const uchar4*>(row_w + (elem_idx >> 1));
        device const half4* x_vec = reinterpret_cast<device const half4*>(X + elem_idx);
        half4 x0 = x_vec[0];
        half4 x1 = x_vec[1];

        float4 w0 = float4(int4(packed & 0x0Fu) - 8);
        float4 w1 = float4(int4((packed >> 4) & 0x0Fu) - 8);

        float4 x_even = float4(x0.x, x0.z, x1.x, x1.z);
        float4 x_odd  = float4(x0.y, x0.w, x1.y, x1.w);

        float sub = dot(w0, x_even) + dot(w1, x_odd);
        acc = fma(sub, scale, acc);
    }

    acc = simd_sum(acc);
    if (lane_id == 0) {
        Y[row] = static_cast<half>(acc);
    }
}

// ============================================================================
// 5. Template Coalesced 1-Row-Per-SIMDgroup (Compile-Time Constant G)
// Eliminates runtime integer division udiv by specializing G at compile time.
// ============================================================================
template <uint G>
inline void gemv_int4_coalesced_1row_impl(
    device const uchar*  W_packed,
    device const half*   Scales,
    device const half*   X,
    device half*         Y,
    uint                 K,
    uint                 N,
    uint2                gid,
    uint                 simd_id,
    uint                 lane_id)
{
    uint row = gid.x * 4 + simd_id;
    if (row >= N) return;

    device const uchar* row_w = W_packed + row * (K >> 1);
    device const half*  row_s = Scales + row * (K / G);

    float acc = 0.0f;

    for (uint elem_idx = lane_id * 8; elem_idx < K; elem_idx += SIMD_SIZE * 8) {
        uint g = elem_idx / G;
        float scale = float(row_s[g]);

        uchar4 packed = *reinterpret_cast<device const uchar4*>(row_w + (elem_idx >> 1));
        device const half4* x_vec = reinterpret_cast<device const half4*>(X + elem_idx);
        half4 x0 = x_vec[0];
        half4 x1 = x_vec[1];

        float4 w0 = float4(int4(packed & 0x0Fu) - 8);
        float4 w1 = float4(int4((packed >> 4) & 0x0Fu) - 8);

        float4 x_even = float4(x0.x, x0.z, x1.x, x1.z);
        float4 x_odd  = float4(x0.y, x0.w, x1.y, x1.w);

        float sub = dot(w0, x_even) + dot(w1, x_odd);
        acc = fma(sub, scale, acc);
    }

    acc = simd_sum(acc);
    if (lane_id == 0) {
        Y[row] = static_cast<half>(acc);
    }
}

// ============================================================================
// 6. Template Coalesced 4-Row-Register-Tiled (Compile-Time Constant G)
// Each SIMDgroup computes 4 rows simultaneously, reusing X in registers.
// ============================================================================
template <uint G>
inline void gemv_int4_coalesced_4row_impl(
    device const uchar*  W_packed,
    device const half*   Scales,
    device const half*   X,
    device half*         Y,
    uint                 K,
    uint                 N,
    uint2                gid,
    uint                 simd_id,
    uint                 lane_id)
{
    uint base_row = gid.x * 8 + simd_id * 4;
    if (base_row >= N) return;

    uint half_K = K >> 1;
    uint num_groups = K / G;

    device const uchar* W_r0 = W_packed + (base_row + 0) * half_K;
    device const uchar* W_r1 = W_packed + (base_row + 1) * half_K;
    device const uchar* W_r2 = W_packed + (base_row + 2) * half_K;
    device const uchar* W_r3 = W_packed + (base_row + 3) * half_K;

    device const half* S_r0 = Scales + (base_row + 0) * num_groups;
    device const half* S_r1 = Scales + (base_row + 1) * num_groups;
    device const half* S_r2 = Scales + (base_row + 2) * num_groups;
    device const half* S_r3 = Scales + (base_row + 3) * num_groups;

    float4 acc = float4(0.0f);

    for (uint elem_idx = lane_id * 8; elem_idx < K; elem_idx += SIMD_SIZE * 8) {
        uint g = elem_idx / G;
        uint byte_idx = elem_idx >> 1;

        device const half4* x_vec = reinterpret_cast<device const half4*>(X + elem_idx);
        half4 x0 = x_vec[0];
        half4 x1 = x_vec[1];
        float4 x_even = float4(x0.x, x0.z, x1.x, x1.z);
        float4 x_odd  = float4(x0.y, x0.w, x1.y, x1.w);

        uchar4 p0 = *reinterpret_cast<device const uchar4*>(W_r0 + byte_idx);
        uchar4 p1 = *reinterpret_cast<device const uchar4*>(W_r1 + byte_idx);
        uchar4 p2 = *reinterpret_cast<device const uchar4*>(W_r2 + byte_idx);
        uchar4 p3 = *reinterpret_cast<device const uchar4*>(W_r3 + byte_idx);

        float dot0 = dot(float4(int4(p0 & 0x0Fu) - 8), x_even) + dot(float4(int4((p0 >> 4) & 0x0Fu) - 8), x_odd);
        float dot1 = dot(float4(int4(p1 & 0x0Fu) - 8), x_even) + dot(float4(int4((p1 >> 4) & 0x0Fu) - 8), x_odd);
        float dot2 = dot(float4(int4(p2 & 0x0Fu) - 8), x_even) + dot(float4(int4((p2 >> 4) & 0x0Fu) - 8), x_odd);
        float dot3 = dot(float4(int4(p3 & 0x0Fu) - 8), x_even) + dot(float4(int4((p3 >> 4) & 0x0Fu) - 8), x_odd);

        acc.x = fma(dot0, float(S_r0[g]), acc.x);
        acc.y = fma(dot1, float(S_r1[g]), acc.y);
        acc.z = fma(dot2, float(S_r2[g]), acc.z);
        acc.w = fma(dot3, float(S_r3[g]), acc.w);
    }

    acc = simd_sum(acc);
    if (lane_id == 0) {
        *reinterpret_cast<device half4*>(Y + base_row) = half4(acc);
    }
}

#define INSTANTIATE_INT4_KERNELS(G) \
kernel void gemv_int4_1row_G##G( \
    device const uchar* W [[buffer(0)]], device const half* S [[buffer(1)]], \
    device const half* X [[buffer(2)]], device half* Y [[buffer(3)]], \
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], \
    uint2 gid [[threadgroup_position_in_grid]], uint simd_id [[simdgroup_index_in_threadgroup]], uint lane_id [[thread_index_in_simdgroup]]) \
{ gemv_int4_coalesced_1row_impl<G>(W, S, X, Y, K, N, gid, simd_id, lane_id); } \
kernel void gemv_int4_4row_G##G( \
    device const uchar* W [[buffer(0)]], device const half* S [[buffer(1)]], \
    device const half* X [[buffer(2)]], device half* Y [[buffer(3)]], \
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], \
    uint2 gid [[threadgroup_position_in_grid]], uint simd_id [[simdgroup_index_in_threadgroup]], uint lane_id [[thread_index_in_simdgroup]]) \
{ gemv_int4_coalesced_4row_impl<G>(W, S, X, Y, K, N, gid, simd_id, lane_id); }

INSTANTIATE_INT4_KERNELS(16)
INSTANTIATE_INT4_KERNELS(24)
INSTANTIATE_INT4_KERNELS(32)
INSTANTIATE_INT4_KERNELS(48)
INSTANTIATE_INT4_KERNELS(64)
INSTANTIATE_INT4_KERNELS(72)
INSTANTIATE_INT4_KERNELS(96)
INSTANTIATE_INT4_KERNELS(128)
INSTANTIATE_INT4_KERNELS(256)

// ============================================================================
// 7. Legacy Group-Strided INT4 Kernel (1 Thread = 1 Group, 1 Row per TG)
// Retained for Ablation 4 to demonstrate the Wave-Quantization & Thread-Starvation
// confounder when num_groups < 128.
// ============================================================================
kernel void gemv_int4_group_strided(
    device const uchar*  W_packed   [[buffer(0)]],
    device const half*   Scales     [[buffer(1)]],
    device const half*   X          [[buffer(2)]],
    device half*         Y          [[buffer(3)]],
    constant uint&       K          [[buffer(4)]],
    constant uint&       N          [[buffer(5)]],
    constant uint&       group_size [[buffer(6)]],
    uint2                gid        [[threadgroup_position_in_grid]],
    uint                 tid        [[thread_index_in_threadgroup]],
    uint                 simd_id    [[simdgroup_index_in_threadgroup]],
    uint                 lane_id    [[thread_index_in_simdgroup]])
{
    uint row = gid.x;
    if (row >= N) return;

    device const uchar* row_w = W_packed + row * (K / 2);
    device const half*  row_s = Scales + row * (K / group_size);
    uint num_groups = K / group_size;

    float acc = 0.0f;

    for (uint g = tid; g < num_groups; g += 128) {
        float scale = float(row_s[g]);
        uint elem_offset = g * group_size;
        uint byte_offset = elem_offset / 2;

        float group_acc = 0.0f;
        for (uint i = 0; i < group_size; i += 8) {
            uint cur_byte = byte_offset + (i / 2);
            device const uchar* b_ptr = row_w + cur_byte;
            device const half*  x_ptr = X + elem_offset + i;

            for (uint j = 0; j < 4; ++j) {
                uchar b = b_ptr[j];
                int n0 = int(b & 0x0F) - 8;
                int n1 = int((b >> 4) & 0x0F) - 8;

                float x0 = float(x_ptr[2 * j]);
                float x1 = float(x_ptr[2 * j + 1]);

                group_acc += float(n0) * x0 + float(n1) * x1;
            }
        }
        acc += group_acc * scale;
    }

    acc = simd_sum(acc);

    threadgroup float tg_acc[8];
    if (lane_id == 0) {
        tg_acc[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < 4; ++s) {
            total += tg_acc[s];
        }
        Y[row] = static_cast<half>(total);
    }
}
