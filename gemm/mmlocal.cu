/**
 * matrix_mul_4096.cu
 *
 * nvcc -O3 -std=c++17 -o matmul_4096 matrix_mul_4096.cu
 *
 * Launches the tiled batched-GEMM kernel supplied by the user on
 * A (1 × 4096 × 4096)  *  B (1 × 4096 × 4096)  →  C (1 × 4096 × 4096)
 *
 * ────────────────────────────────────────────────────────────────────
 * The code shows only the minimum boiler-plate needed:
 *   • allocate & fill host data
 *   • copy to GPU
 *   • prepare shape / stride metadata
 *   • launch the kernel
 *   • copy result back & sanity-check against cuBLAS (optional)
 */

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>

#define TILE 32 // same tile size the kernel expects
#define CHECK(call)                                                       \
    do                                                                    \
    {                                                                     \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                             \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

// ───────────────────────────────── device helpers ────────────────
__device__ __forceinline__ int index_to_position(const int idx[3], const int strides[3], int dims)
{
    int pos = 0;
#pragma unroll
    for (int d = 0; d < dims; ++d)
        pos += idx[d] * strides[d];
    return pos;
}

// ──────────────────────────────── user kernel ────────────────────
__global__ void MatrixMultiplyKernel(
    float *out,
    const int *out_shape,
    const int *out_strides,
    const float *a_storage,
    const int *a_shape,
    const int *a_strides,
    const float *b_storage,
    const int *b_shape,
    const int *b_strides)
{
    __shared__ float a_shared[TILE][TILE + 4];
    __shared__ float b_shared[TILE][TILE + 4];

    int batch = blockIdx.z;
    int a_batch_stride = a_shape[0] > 1 ? a_strides[0] : 0;
    int b_batch_stride = b_shape[0] > 1 ? b_strides[0] : 0;
    int i = blockIdx.x;
    int j = blockIdx.y;
    int aIndex[3];
    int bIndex[3];
    int cIndex[3];
    aIndex[0] = batch;
    bIndex[0] = batch;
    cIndex[0] = batch;
    cIndex[1] = i * blockDim.x + threadIdx.x;
    cIndex[2] = j * blockDim.y*4;

    int a_local_shape[3];
    int a_local_strides[3];
    int b_local_shape[3];
    int b_local_strides[3];
    int out_local_shape[3];
    int out_local_strides[3];

    for (int idx = 0; idx < 3; idx++)
    {
        a_local_shape[idx] = a_shape[idx];
        a_local_strides[idx] = a_strides[idx];
        b_local_shape[idx] = b_shape[idx];
        b_local_strides[idx] = b_strides[idx];
        out_local_shape[idx] = out_shape[idx];
        out_local_strides[idx] = out_strides[idx];
    }

    aIndex[1] = cIndex[1];
    aIndex[2] = 0;
    int baseAIndex = index_to_position(aIndex, a_local_strides, 3);
    float4 accum = {0.f,0.f,0.f,0.f};
    int threadY4Idx = threadIdx.y;

    for (int k = 0; k < a_local_shape[2]; k += TILE)
    {
        float4 *a_shared4_ptr = reinterpret_cast<float4 *>(&a_shared[threadIdx.x][0]);

        float4 const *a_storage4_ptr = reinterpret_cast<float4 const *>(&a_storage[baseAIndex + k]);

        const bool inRangeA{aIndex[1] < a_local_shape[1] && k + threadY4Idx*4 < a_local_shape[2]};
        a_shared4_ptr[threadY4Idx] = inRangeA ? a_storage4_ptr[threadY4Idx] : float4{0.f, 0.f, 0.f, 0.f};

        bIndex[1] = k + threadIdx.x;
        bIndex[2] = j * blockDim.y;
        int baseBIndex = index_to_position(bIndex, b_local_strides, 3);
        float4 const *b_storage4_ptr = reinterpret_cast<float4 const *>(&b_storage[baseBIndex]);
        float4 *b_shared4_ptr = reinterpret_cast<float4 *>(&b_shared[threadIdx.x][0]);
        const bool inRangeB{bIndex[1] < b_local_shape[1] && bIndex[2]+threadY4Idx*4 < b_local_shape[2]};
        b_shared4_ptr[threadY4Idx] = (inRangeB) ? b_storage4_ptr[threadY4Idx] : float4{0.f, 0.f, 0.f, 0.f};
        __syncthreads();

        #pragma unroll
        for (int tileIdx = 0; tileIdx < TILE; tileIdx+=4)
        {
            float4 avec = a_shared4_ptr[tileIdx/4];
            float4 bvectile0 = *reinterpret_cast<float4 *>(&b_shared[tileIdx][threadIdx.y*4]);
            float4 bvectile1 = *reinterpret_cast<float4 *>(&b_shared[tileIdx+1][threadIdx.y*4]);
            float4 bvectile2 = *reinterpret_cast<float4 *>(&b_shared[tileIdx+2][threadIdx.y*4]);
            float4 bvectile3 = *reinterpret_cast<float4 *>(&b_shared[tileIdx+3][threadIdx.y*4]);
            accum.x += avec.x * bvectile0.x + avec.y * bvectile1.x + avec.z * bvectile2.x + avec.w * bvectile3.x;
            accum.y += avec.x * bvectile0.y + avec.y * bvectile1.y + avec.z * bvectile2.y + avec.w * bvectile3.y;
            accum.z += avec.x * bvectile0.z + avec.y * bvectile1.z + avec.z * bvectile2.z + avec.w * bvectile3.z;
            accum.w += avec.x * bvectile0.w + avec.y * bvectile1.w + avec.z * bvectile2.w + avec.w * bvectile3.w;
            // printf("avec %f %f %f %f\n", avec.x, avec.y, avec.z, avec.w);
            // printf("bvectile0 %f %f %f %f\n", bvectile0.x, bvectile0.y, bvectile0.z, bvectile0.w);
            // printf("bvectile1 %f %f %f %f\n", bvectile1.x, bvectile1.y, bvectile1.z, bvectile1.w);
            // printf("bvectile2 %f %f %f %f\n", bvectile2.x, bvectile2.y, bvectile2.z, bvectile2.w);
            // printf("bvectile3 %f %f %f %f\n", bvectile3.x, bvectile3.y, bvectile3.z, bvectile3.w);
        }
    }
    if (cIndex[1] < out_local_shape[1] && cIndex[2]+threadY4Idx*4 < out_local_shape[2])
    {
        int linearOutIndex = index_to_position(cIndex, out_local_strides, 3);
        float4* outPtr4 = reinterpret_cast<float4*>(&out[linearOutIndex]);
        // printf("idx: %d %f %f %f %f\n", linearOutIndex+threadY4Idx*4, accum.x, accum.y, accum.z, accum.w);
        outPtr4[threadY4Idx] = accum;   
    }
    /// END ASSIGN1_2
}


// ──────────────────────────────── host main ───────────────────────
int main()
{
    int size = 4096;
    int B = 1, M = size, N = size, P = size;
    const size_t bytesA = size_t(B) * M * N * sizeof(float);
    const size_t bytesB = size_t(B) * N * P * sizeof(float);
    const size_t bytesC = size_t(B) * M * P * sizeof(float);

    // Allocate pinned host buffers
    float *hA, *hB, *hC;
    CHECK(cudaMallocHost(&hA, bytesA));
    CHECK(cudaMallocHost(&hB, bytesB));
    CHECK(cudaMallocHost(&hC, bytesC));

    // Fill with some deterministic data
    for (size_t i = 0; i < (bytesA / sizeof(float)); ++i)
        hA[i] = 1.f;
    for (size_t i = 0; i < (bytesB / sizeof(float)); ++i)
        hB[i] = 1.f;

    // Allocate device buffers
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytesA));
    CHECK(cudaMalloc(&dB, bytesB));
    CHECK(cudaMalloc(&dC, bytesC));

    CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    // shape & stride metadata ---------------------------------------
    int h_a_shape[3] = {B, M, N};
    int h_b_shape[3] = {B, N, P};
    int h_out_shape[3] = {B, M, P};

    int h_a_strides[3] = {M * N, N, 1};
    int h_b_strides[3] = {N * P, P, 1};
    int h_out_strides[3] = {M * P, P, 1};

    int *d_a_shape, *d_b_shape, *d_out_shape;
    int *d_a_strides, *d_b_strides, *d_out_strides;
    CHECK(cudaMalloc(&d_a_shape, 3 * sizeof(int)));
    CHECK(cudaMalloc(&d_b_shape, 3 * sizeof(int)));
    CHECK(cudaMalloc(&d_out_shape, 3 * sizeof(int)));
    CHECK(cudaMalloc(&d_a_strides, 3 * sizeof(int)));
    CHECK(cudaMalloc(&d_b_strides, 3 * sizeof(int)));
    CHECK(cudaMalloc(&d_out_strides, 3 * sizeof(int)));

    CHECK(cudaMemcpy(d_a_shape, h_a_shape, 3 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b_shape, h_b_shape, 3 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_out_shape, h_out_shape, 3 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_a_strides, h_a_strides, 3 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b_strides, h_b_strides, 3 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_out_strides, h_out_strides, 3 * sizeof(int), cudaMemcpyHostToDevice));

    // Kernel launch configuration -----------------------------------
    dim3 block(TILE, TILE/4, 1);
    dim3 grid((M + TILE - 1) / TILE,
              (P + TILE - 1) / TILE,
              B);

    MatrixMultiplyKernel<<<grid, block>>>(
        dC, d_out_shape, d_out_strides,
        dA, d_a_shape, d_a_strides,
        dB, d_b_shape, d_b_strides);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Copy result back
    CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    // (optional) quick sanity check – C should be all N when A,B filled with 1s
    double max_err = 0.;
    for (int i = 0; i < (bytesC / sizeof(float)); ++i)
    {
        // printf("(%d %f),", i, hC[i]);
        max_err = fmax(max_err, fabs(hC[i] - float(N)));
    }
    printf("max error = %g (should be 0)\n", max_err);

    // clean up
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    CHECK(cudaFree(d_a_shape));
    CHECK(cudaFree(d_b_shape));
    CHECK(cudaFree(d_out_shape));
    CHECK(cudaFree(d_a_strides));
    CHECK(cudaFree(d_b_strides));
    CHECK(cudaFree(d_out_strides));
    CHECK(cudaFreeHost(hA));
    CHECK(cudaFreeHost(hB));
    CHECK(cudaFreeHost(hC));
    return 0;
}