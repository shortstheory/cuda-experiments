#include <math.h>

#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>
#include <cstdlib>
#include <ctime>
#include <cooperative_groups.h>
#include <iostream>
#include <string>  // Required for std::stoi()
constexpr int TILE = 512;

namespace cg = cooperative_groups;
using namespace cooperative_groups; // or...
using cooperative_groups::thread_group; // etc.
constexpr int PARTITION_SIZE = 32;

__global__ void myBlockScan(float *data, int n) {
    __shared__ float sharedData[1024];
    int idx = threadIdx.x;
    sharedData[idx] = 0.f;
    if (idx < n)
    {
        sharedData[idx] = data[idx];
    }
    __syncthreads();

    for (int i = 1; i < n; i*=2)
    {
        if (idx > i)
        {
            sharedData[idx] = sharedData[idx] + sharedData[idx-i];
        }
        __syncthreads();
    }

    if (idx < n)
    {
        data[idx] = sharedData[idx];
    }
}

__global__ void myShuffleScan(float *data, int n) {
    __shared__ float sharedData[1024/PARTITION_SIZE];
    // __shared__ float sharedData2[1024/PARTITION_SIZE];

    int idx = threadIdx.x;
    float threadValue = 0.f;
    if (idx < n)
    {
        threadValue = data[idx];
    }

    auto tile = cg::tiled_partition<PARTITION_SIZE>(this_thread_block());

    #pragma unroll
    for (int i = 1; i < PARTITION_SIZE; i *= 2) {
        float temp = tile.shfl_up(threadValue, i);
        if (idx % PARTITION_SIZE >= i)
        {
            threadValue += temp;
        }
    }
    if (tile.thread_rank()+1 == PARTITION_SIZE)
    {
        sharedData[idx/PARTITION_SIZE] = threadValue;
    }
    __syncthreads();

    if (idx < PARTITION_SIZE)
    {
        #pragma unroll
        for (int i = 1; i < PARTITION_SIZE; i *= 2) {
            float temp = tile.shfl_up(sharedData[tile.thread_rank()], i);
            if (tile.thread_rank() % PARTITION_SIZE >= i)
            {
                sharedData[tile.thread_rank()] += temp;
            }
        }
    }
    __syncthreads();

    if (idx > PARTITION_SIZE)
        threadValue += sharedData[idx/PARTITION_SIZE-1];

    if (idx < n)
    {
        data[idx] = threadValue;
    }
}


int main(int argc, char **argv)
{
    int n = 1000;
    if (argc > 1) 
    {
        n = std::stoi(argv[1]);
    }
    std::cout << "N: " << n << std::endl;
    float* data = new float[n];
    float* gpuResultCpu = new float[n];
    float* gpuData;
    cudaError_t error = cudaMalloc(&gpuData, n*sizeof(float));
    std::cout << cudaGetErrorString(error) << std::endl;
    for (int i = 0; i < n; i++)
    {
        data[i] = i;
    }

    cudaMemcpy(gpuData, data, n*sizeof(float), cudaMemcpyHostToDevice);
    myBlockScan<<<1,1024>>>(gpuData,n);

    cudaMemcpy(gpuData, data, n*sizeof(float), cudaMemcpyHostToDevice);
    myShuffleScan<<<1,1024>>>(gpuData,n);

    cudaMemcpy(gpuResultCpu, gpuData, n*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 1; i < n; i++)
    {
        data[i] = data[i] + data[i-1];
    }
    for (int i = 0; i < n; i++)
    {
        std::cout << i << " " << data[i] << " " << gpuResultCpu[i] << std::endl;
    }
    // std::cout << std::endl;
    // for (int i = 0; i < n; i++)
    // {
    //     std::cout << i << " " <<  << " " << std::endl;
    // }

    return 0;
}