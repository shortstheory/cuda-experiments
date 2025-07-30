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
    /**
     * Multiply two (compact) matrices into an output (also comapct) matrix. Matrix a and b are both in a batch
     * format, with shape [batch_size, m, n], [batch_size, n, p].
     * Requirements:
     * - All data must be first moved to shared memory.
     * - Only read each cell in a and b once.
     * - Only write to global memory once per kernel.
     * There is guarantee that a_shape[0] == b_shape[0], a_shape[2] == b_shape[1],
     * and out_shape[0] == a_shape[0], out_shape[1] == b_shape[1]
     *
     * Args:
     *   out: compact 1D array of size batch_size x m x p to write the output to
     *   out_shape: shape of the output array
     *   out_strides: strides of the output array
     *   a_storage: compact 1D array of size batch_size x m x n
     *   a_shape: shape of the a array
     *   a_strides: strides of the a array
     *   b_storage: comapct 2D array of size batch_size x n x p
     *   b_shape: shape of the b array
     *   b_strides: strides of the b array
     *
     * Returns:
     *   None (Fills in out array)
     */

    __shared__ float a_shared[TILE][TILE + 4];
    __shared__ float b_shared[TILE][TILE + 4];

    // In each block, we will compute a batch of the output matrix
    // All the threads in the block will work together to compute this batch
    int batch = blockIdx.z;
    int a_batch_stride = a_shape[0] > 1 ? a_strides[0] : 0;
    int b_batch_stride = b_shape[0] > 1 ? b_strides[0] : 0;

    /// BEGIN ASSIGN1_2
    /// TODO
    // Hints:
    // 1. Compute the row and column of the output matrix this block will compute
    // 2. Compute the position in the output array that this thread will write to
    // 3. Iterate over tiles of the two input matrices, read the data into shared memory
    // 4. Synchronize to make sure the data is available to all threads
    // 5. Compute the output tile for this thread block
    // 6. Synchronize to make sure all threads are done computing the output tile for (row, col)
    // 7. Write the output to global memory

    int i = blockIdx.x;
    int j = blockIdx.y;
    int aIndex[3];
    int bIndex[3];
    int cIndex[3];
    aIndex[0] = batch;
    bIndex[0] = batch;
    cIndex[0] = batch;
    cIndex[1] = i * blockDim.x + threadIdx.x;
    cIndex[2] = j * blockDim.y + threadIdx.y;
    float accum = 0.f;

    int a_local_shape[3];
    int a_local_strides[3];
    int b_local_shape[3];
    int b_local_strides[3];
    int out_local_shape[3];
    int out_local_strides[3];

    for (int i = 0; i < 3; i++)
    {
        a_local_shape[i] = a_shape[i];
        a_local_strides[i] = a_strides[i];
        b_local_shape[i] = b_shape[i];
        b_local_strides[i] = b_strides[i];
        out_local_shape[i] = out_shape[i];
        out_local_strides[i] = out_strides[i];
    }

    aIndex[1] = cIndex[1];
    bIndex[2] = cIndex[2];
    aIndex[2] = 0;
    int baseAIndex = index_to_position(aIndex, a_local_strides, 3);

    for (int k = 0; k < a_local_shape[2]; k += TILE)
    {
        int threadY4Idx = threadIdx.y / 4;
        aIndex[2] = k + threadY4Idx;
        bIndex[1] = k + threadIdx.x;

        float4 *a_shared4_ptr = reinterpret_cast<float4 *>(&a_shared[threadIdx.x][0]);
        float4 const *a_storage4_ptr = reinterpret_cast<float4 const *>(&a_storage[baseAIndex+k]);

        if (aIndex[1] < a_local_shape[1] && aIndex[2] < a_local_shape[2])
        {
            if (threadIdx.y % 4 == 0)
            {
                a_shared4_ptr[threadY4Idx] = a_storage4_ptr[threadY4Idx];
            }
        }
        else
        {
            a_shared4_ptr[threadY4Idx] = float4{0.f, 0.f, 0.f, 0.f};
        }
        if (bIndex[1] < b_local_shape[1] && bIndex[2] < b_local_shape[2])
        {
            int linearBIndex = index_to_position(bIndex, b_local_strides, 3);
            b_shared[threadIdx.y][threadIdx.x] = b_storage[linearBIndex];
        }
        else
        {
            b_shared[threadIdx.y][threadIdx.x] = 0.f;
        }
        __syncthreads();

// float4* a_vec = reinterpret_cast<float4*>(a_shared[threadIdx.x]);
#pragma unroll
        for (int tileIdx = 0; tileIdx < TILE / 4; tileIdx++)
        {
            // float4 a_val = a_vec[tileIdx];
            accum += a_shared[threadIdx.x][tileIdx] * b_shared[threadIdx.y][tileIdx] + a_shared[threadIdx.x][tileIdx + 1] * b_shared[threadIdx.y][tileIdx + 1] + a_shared[threadIdx.x][tileIdx + 2] * b_shared[threadIdx.y][tileIdx + 2] + a_shared[threadIdx.x][tileIdx + 3] * b_shared[threadIdx.y][tileIdx + 3];
        }
    }
    if (cIndex[1] < out_local_shape[1] && cIndex[2] < out_local_shape[2])
    {
        int linearOutIndex = index_to_position(cIndex, out_local_strides, 3);
        out[linearOutIndex] = accum; // c_shared[threadIdx.x][threadIdx.y];
    }
    /// END ASSIGN1_2
}

// ──────────────────────────────── host main ───────────────────────
int main()
{
    constexpr int B = 1, M = 4096, N = 4096, P = 4096;
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
    dim3 block(TILE, TILE, 1);
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
    for (size_t i = 0; i < (bytesC / sizeof(float)); ++i)
        max_err = fmax(max_err, fabs(hC[i] - float(N)));
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