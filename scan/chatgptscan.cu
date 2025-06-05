// decoupled_lookback_scan.cu
// ------------------------------------------------------------
// A minimal, standalone implementation of an *exclusive* prefix-sum
// (scan) using the **decoupled look-back** technique described by
// Merrill & Garland (GPU Technology Conference 2016).
//
//  • Each thread processes `ITEMS_PER_THREAD` input elements.
//  • Each thread-block produces a block-aggregate and uses a global
//    array of flags + aggregates to acquire the running prefix of
//    all previous blocks **without a separate kernel launch**.
//
// Compile & run (compute ≥ 6.0):
//     nvcc -arch=sm_60 -O3 decoupled_lookback_scan.cu -o scan && ./scan 1048576
// ------------------------------------------------------------
#include <cstdio>
#include <cuda_runtime.h>

//--------------------------------------------------------------------
// Helpers
//--------------------------------------------------------------------
#define CUDA_CHECK(expr)                                    \
    do {                                                   \
        cudaError_t _err = (expr);                         \
        if (_err != cudaSuccess) {                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n",   \
                    cudaGetErrorString(_err), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                       \
        }                                                  \
    } while (0)

template <typename T>
__device__ __forceinline__ T warp_exclusive_scan(T val) {
    // 32-wide warp, inclusive then shift.
    for (int offset = 1; offset < 32; offset <<= 1) {
        T n = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % 32 >= offset) val += n;
    }
    return val - __shfl_sync(0xffffffff, val, 31); // convert inclusive→exclusive
}

//--------------------------------------------------------------------
// Kernel parameters
//--------------------------------------------------------------------
constexpr int BLOCK_THREADS       = 256;
constexpr int ITEMS_PER_THREAD    = 4;   // so block handles 1024 elements

//--------------------------------------------------------------------
// Global arrays used by decoupled look-back
//--------------------------------------------------------------------
struct BlockPrefix {
    float  inclusive;   // aggregate of this block (inclusive scan of block sums)
    int    ready;       // 0 → waiting, 1 → written (BLOCK_SUM_READY), 2 → done
};

//--------------------------------------------------------------------
// Kernel
//--------------------------------------------------------------------
__global__ void scan_decoupled_lookback(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        BlockPrefix*  __restrict__ prefix,
                                        int n)
{
    const int tid    = threadIdx.x;
    const int bid    = blockIdx.x;
    const int block_first = (bid * BLOCK_THREADS * ITEMS_PER_THREAD);

    //----------------------------------------------------------------
    // 1. Load & local exclusive scan (in registers)
    //----------------------------------------------------------------
    float thread_data[ITEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = block_first + tid + i * BLOCK_THREADS;
        thread_data[i] = (idx < n) ? in[idx] : 0.f;
    }

    // In-register exclusive scan per thread (serial)
    float running = 0.f;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        float x = thread_data[i];
        thread_data[i] = running;
        running += x;
    }

    // Warp exclusive scan of each thread's block-sum (running)
    float block_sum = running;                          // sum of this thread's items
    float scan_base = warp_exclusive_scan(block_sum);   // prefix of thread sums

    // Broadcast warp-level sums into shared memory for final upsweep
    __shared__ float warp_sums[BLOCK_THREADS/32];
    if (tid % 32 == 31) warp_sums[tid/32] = scan_base + block_sum;
    __syncthreads();

    // Last warp performs exclusive scan of warp aggregates
    if (tid/32 == 0) {
        float warp_prefix = warp_exclusive_scan(warp_sums[tid%32]);
        warp_sums[tid%32] = warp_prefix;
    }
    __syncthreads();

    // Each thread's base = warp_prefix + intra-warp prefix
    scan_base += warp_sums[tid/32];

    // Add base to thread_data
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) thread_data[i] += scan_base;

    //----------------------------------------------------------------
    // 2. Publish block aggregate & obtain prefix from previous blocks
    //----------------------------------------------------------------
    float block_aggregate = warp_sums[(BLOCK_THREADS/32)-1] +
                            __shfl_sync(0xffffffff, block_sum, 31);

    if (tid == 0) {
        prefix[bid].inclusive = block_aggregate;
        __threadfence();                 // flush before flag
        prefix[bid].ready     = 1;       // BLOCK_SUM_READY
    }

    // look-back phase – performed by thread 0, then broadcast via shared mem
    __shared__ float block_base;
    if (tid == 0) {
        float accum = 0.f;
        int   look  = bid - 1;
        while (look >= 0) {
            int state = prefix[look].ready;
            if (state == 1) {            // READY
                accum += prefix[look].inclusive;
                look--;
            }
            else if (state == 2) {      // DONE – we can jump out
                accum += prefix[look].inclusive;
                break;
            }
        }
        // mark our block DONE and store inclusive prefix for later blocks
        prefix[bid].inclusive += accum;
        __threadfence();
        prefix[bid].ready = 2;           // BLOCK_SUM_DONE
        block_base = accum;              // exclusive base for this block
    }
    __syncthreads();

    //----------------------------------------------------------------
    // 3. Add block_base to every element and write out
    //----------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = block_first + tid + i * BLOCK_THREADS;
        if (idx < n) out[idx] = thread_data[i] + block_base;
    }
}

//--------------------------------------------------------------------
// Host test-driver
//--------------------------------------------------------------------
void reference_scan(float* h, int n) {
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        float x = h[i];
        h[i] = sum;            // exclusive
        sum += x;
    }
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 1<<20; // default 1M
    size_t bytes = n * sizeof(float);

    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) h_in[i] = 1.f;      // easy to verify

    float *d_in, *d_out;  CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // prefix array – one entry per block
    int numBlocks = (n + BLOCK_THREADS*ITEMS_PER_THREAD - 1) /
                    (BLOCK_THREADS*ITEMS_PER_THREAD);
    BlockPrefix* d_prefix;  CUDA_CHECK(cudaMalloc(&d_prefix, numBlocks * sizeof(BlockPrefix)));
    CUDA_CHECK(cudaMemset(d_prefix, 0, numBlocks * sizeof(BlockPrefix)));

    // launch
    scan_decoupled_lookback<<<numBlocks, BLOCK_THREADS>>>(d_in, d_out, d_prefix, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // reference & verify
    reference_scan(h_in, n);
    int errors = 0;
    for (int i = 0; i < n; ++i) if (fabs(h_in[i] - h_out[i]) > 1e-5) { errors++; break; }
    printf("%s!  (n = %d)\n", errors ? "Mismatch" : "Success", n);

    // cleanup
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_prefix);
    free(h_in); free(h_out);
    return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
