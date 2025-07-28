#include <cuda_runtime.h>
#include <stdio.h>

__global__ void memory_stress_slow(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Memory access loop, minimal computation
        for (int i = 0; i < 1; ++i) {
            data[idx] = data[idx] * 1.00001f;
        }
    }
}

int main() {
    const int N = 1 << 24;  // ~16 million elements
    size_t size = N * sizeof(float);
    float* d_data;

    cudaMalloc(&d_data, size);
    cudaMemset(d_data, 0, size);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    memory_stress_slow<<<grid, block>>>(d_data, N);

    cudaDeviceSynchronize();
    cudaFree(d_data);
    return 0;
}