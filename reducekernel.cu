#include <math.h>

#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>
#include <cstdlib>
#include <ctime>
#include <cooperative_groups.h>
#include <iostream>
#include <string>  // Required for std::stoi()
constexpr int TILE = 128;
__global__ void simpleReduce1(float* out, float* data, int size)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    float threadSum = 0.0;
    if (idx < size)
    {
        threadSum = data[idx];
    }
    __shared__ float s;
    if (threadIdx.x == 0)
    {
        s = 0.f;
    }
    __syncthreads();
    for (int i = 0; i < blockDim.x; i++)
    {
        if (threadIdx.x == i)
        {
            s += threadSum;
            // printf("threadIdx.x %d Val %f s %f\n", threadIdx.x, threadSum, s);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        // atomicAdd(out, s);
        data[blockDim.x*blockIdx.x] = s;
    }
}

__global__ void simpleReduce14float(float* out, float* data, int size)
{
    float4* data_f4 = reinterpret_cast<float4*>(data);
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    float threadSum = 0.0;
    float4 val;
    if (idx < size/4)
    {
        val = data_f4[idx];
        threadSum = val.x + val.y + val.z + val.w;
    }
    __shared__ float s;
    if (threadIdx.x == 0)
    {
        s = 0.f;
    }
    __syncthreads();
    for (int i = 0; i < blockDim.x; i++)
    {
        __syncthreads();
        if (threadIdx.x == i)
        {
            s += threadSum;
            // printf("threadIdx.x %d Val %f s %f\n", threadIdx.x, threadSum, s);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        // atomicAdd(out, s);
        data[blockDim.x*blockIdx.x] = s;
    }
}


__global__ void simpleReduce2(float* out, float* data, int size)
{
    __shared__ float buffer[TILE];
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        buffer[threadIdx.x] = data[idx];
    } else {
        buffer[threadIdx.x] = 0.f;
    }
    __syncthreads();
    for (int s = 1; s < blockDim.x; s++)
    {
        if (threadIdx.x == 0)
        {
            buffer[0] += buffer[s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        // atomicAdd(out, buffer[0]);
        data[blockDim.x*blockIdx.x] = buffer[0];
    }
}

namespace cg = cooperative_groups;
using namespace cooperative_groups; // or...
using cooperative_groups::thread_group; // etc.
constexpr int PARTITION_SIZE = 32;
__global__ void bigReduce(float* out, float* data, int size)
{
    float4* data_f4 = reinterpret_cast<float4*>(data);
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    float threadSum = 0.0;
    float4 val;
    if (idx < size/4)
    {
        val = data_f4[idx];
        threadSum = val.x + val.y + val.z + val.w;
    }
    auto tile32 = cg::tiled_partition<PARTITION_SIZE>(this_thread_block());
    #pragma unroll
    for (int i = PARTITION_SIZE/2; i > 0; i /= 2) {
        // only template version has it wtf
        threadSum += tile32.shfl_down(threadSum, i);
    }
    constexpr int SIZE =TILE/PARTITION_SIZE;
    __shared__ float threadBuffer[SIZE];
    if (tile32.thread_rank() == 0)
    {
        threadBuffer[tile32.meta_group_rank()] = threadSum;
    }
    __syncthreads();
    float blockVal=0.0;
    if (threadIdx.x == 0)
    {
        #pragma unroll
        for (size_t i = 0; i < SIZE; i++)
        {
            blockVal += threadBuffer[i];
        }
        data[blockIdx.x] = blockVal;
        // atomicAdd(out, blockVal);  
    }
}   

__global__ void bigReduceNoFloat4(float* out, float* data, int size)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    float threadSum = 0.0;
    if (idx < size)
    {
        threadSum = data[idx];
    }
    auto tile32 = cg::tiled_partition<PARTITION_SIZE>(this_thread_block());
    #pragma unroll
    for (int i = PARTITION_SIZE/2; i > 0; i /= 2) {
        // only template version has it wtf
        threadSum += tile32.shfl_down(threadSum, i);
    }
    constexpr int SIZE =TILE/PARTITION_SIZE;
    __shared__ float threadBuffer[SIZE];
    if (tile32.thread_rank() == 0)
    {
        threadBuffer[tile32.meta_group_rank()] = threadSum;
    }
    __syncthreads();
    float blockVal=0.0;
    if (threadIdx.x == 0)
    {
        #pragma unroll
        for (size_t i = 0; i < SIZE; i++)
        {
            blockVal += threadBuffer[i];
        }
        data[blockDim.x*blockIdx.x] = blockVal;
        // atomicAdd(out, blockVal);  
    }
}   


int main(int argc, char **argv)
{
    
    int n = 1000*1000;
    if (argc > 1) 
    {
        n = std::stoi(argv[1]);
    }
    std::cout << "N: " << n << std::endl;
    float* data = new float[n];
    float* gpuData;
    float* gpuRes;
    float gpuResultOnCpu;
    cudaMallocHost(&gpuData, n*sizeof(float));
    cudaMallocHost(&gpuRes, sizeof(float));
    cudaMemset(gpuRes, 0, sizeof(float));
    double res = 0.f;
    for (int i = 0; i < n; i++)
    {
        data[i] = i;
        // data[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    cudaMemcpy(gpuData, data, n*sizeof(float), cudaMemcpyHostToDevice);
    
    int numBlocks = n/TILE+1;
    
    // simpleReduce1<<<numBlocks, TILE>>>(gpuRes, gpuData, n);
    // cudaMemcpy(&gpuResultOnCpu, gpuRes, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "Gpu1: " << gpuResultOnCpu << std::endl;

    // cudaMemset(gpuRes, 0, sizeof(float));
    // simpleReduce14float<<<n/(4*TILE)+1, TILE>>>(gpuRes, gpuData, n);
    // cudaMemcpy(&gpuResultOnCpu, gpuRes, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "Gpu2: " << gpuResultOnCpu << std::endl;

    // cudaMemset(gpuRes, 0, sizeof(float));
    // simpleReduce2<<<numBlocks,TILE>>>(gpuRes, gpuData, n);
    // cudaMemcpy(&gpuResultOnCpu, gpuRes, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "Gpu3: " << gpuResultOnCpu << std::endl;

    cudaMemset(gpuRes, 0, sizeof(float));
    bigReduce<<<n/(4*TILE)+1, TILE>>>(gpuRes, gpuData, n);
    cudaMemcpy(&gpuResultOnCpu, gpuRes, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Gpu4: " << gpuResultOnCpu << std::endl;

    // cudaMemset(gpuRes, 0, sizeof(float));
    // bigReduceNoFloat4<<<numBlocks, TILE>>>(gpuRes, gpuData, n);
    // cudaMemcpy(&gpuResultOnCpu, gpuRes, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "Gpu5: " << gpuResultOnCpu << std::endl;


    for (int i = 0; i < n; i++)
    {
        res += data[i];
    }
    std::cout << "Cpu: " << res << std::endl;
    return 0;
}