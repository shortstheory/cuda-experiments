#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector> 
#include <numeric>
using namespace std;
///////////////////////////////////////////////////////////////////////////////
//  Kernel: naïve per-block reduction with no unrolling or warp-shuffle.
//          – Each thread copies one element to shared memory.
//          – A simple (and inefficient) modulo-stride loop halves active
//            threads each pass.
//          – Block result is added to a single global sum with atomicAdd.
///////////////////////////////////////////////////////////////////////////////
__global__ void reduce_naive(const float *g_in, float *g_out, int n)
{
    extern __shared__ float sdata[];               // dynamic shared memory

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockDim.x * blockIdx.x + tid;

    // Load one element per thread (or 0 if out-of-bounds)
    sdata[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();

    // Naïve reduction: stride = 1,2,4,…   (inefficient thread utilisation)
    for (unsigned int s = 1; s < blockDim.x; s++)
    {
        if (tid == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads();                          // keep shared mem coherent
    }

    // Thread 0 holds this block’s sum → accumulate into global result
    if (tid == 0)
        atomicAdd(g_out, sdata[0]);
}

///////////////////////////////////////////////////////////////////////////////
//  Simple driver code
///////////////////////////////////////////////////////////////////////////////
int main()
{
    const int N         = 1000*1000;                // 1 048 576 elements
    const int TPB       = 256;                    // threads per block
    const int NUMBLOCKS = (N + TPB - 1) / TPB;

    // Host data
    std::vector<float> h_in(N);
    std::iota(h_in.begin(), h_in.end(), 1.0f);     // 1,2,3,…

    // Device buffers
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    // Launch (shared-mem size = TPB * sizeof(float))
    reduce_naive<<<NUMBLOCKS, TPB, TPB * sizeof(float)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // Fetch GPU result
    float gpuSum;
    cudaMemcpy(&gpuSum, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // CPU reference
    double cpuSum = 0.0;
    for (float v : h_in) cpuSum += v;

    printf("CPU sum = %.0f\nGPU sum = %.0f\n", cpuSum, gpuSum);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}