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
        atomicAdd(out, blockVal);  
    }
}   

__global__ void bigReduceBlockLoad(float* out, float* data, int size)
{
    constexpr int ELEMENTS_PER_THREAD = TILE/PARTITION_SIZE;

    float* dataPtr = data + blockDim.x*blockIdx.x*ELEMENTS_PER_THREAD;
    float threadSum = 0.0;
    typedef cub::BlockLoad<float, TILE, ELEMENTS_PER_THREAD,
                        cub::BLOCK_LOAD_VECTORIZE>
    BlockLoad;
    __shared__ typename BlockLoad::TempStorage ts_load;
    float mval[ELEMENTS_PER_THREAD];
    // printf("Starting kernel\n");

    BlockLoad(ts_load).Load(dataPtr, mval, size-  blockDim.x*blockIdx.x*ELEMENTS_PER_THREAD, 0);  
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++)
    {
        // printf("threadIdx.x %d i %d mval %f\n", threadIdx.x, i, mval[i]); 
        threadSum += mval[i];
    }

    auto tile32 = cg::tiled_partition<PARTITION_SIZE>(this_thread_block());
    #pragma unroll
    for (int i = PARTITION_SIZE/2; i > 0; i /= 2) {
        threadSum += tile32.shfl_down(threadSum, i);
    }
    __shared__ float threadBuffer[ELEMENTS_PER_THREAD];
    if (tile32.thread_rank() == 0)
    {
        threadBuffer[tile32.meta_group_rank()] = threadSum;
    }
    __syncthreads();
    float blockVal=0.0;
    if (threadIdx.x == 0)
    {
        #pragma unroll
        for (size_t i = 0; i < ELEMENTS_PER_THREAD; i++)
        {
            blockVal += threadBuffer[i];
        }
        atomicAdd(out, blockVal);  
    }
}   

template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
 if (blockSize >= 64)
   sdata[tid] += sdata[tid + 32];
 if (blockSize >= 32)
   sdata[tid] += sdata[tid + 16];
 if (blockSize >= 16)
   sdata[tid] += sdata[tid + 8];
 if (blockSize >= 8)
   sdata[tid] += sdata[tid + 4];
 if (blockSize >= 4)
   sdata[tid] += sdata[tid + 2];
 if (blockSize >= 2)
   sdata[tid] += sdata[tid + 1];
}


template <unsigned int blockSize>
__global__ void reduce6(float *g_odata, float *g_idata, unsigned int n) {
 extern __shared__ float sdata[];
 unsigned int tid = threadIdx.x;
 unsigned int i = blockIdx.x * (blockSize * 2) + tid;
 unsigned int gridSize = blockSize * 2 * gridDim.x;
 sdata[tid] = 0;
 while (i < n) {
   sdata[tid] += g_idata[i] + g_idata[i + blockSize];
   i += gridSize;
 }
 __syncthreads();
 if (blockSize >= 512) {
   if (tid < 256) {
     sdata[tid] += sdata[tid + 256];
   }
   __syncthreads();
 }
 if (blockSize >= 256) {
   if (tid < 128) {
     sdata[tid] += sdata[tid + 128];
   }
   __syncthreads();
 }
 if (blockSize >= 128) {
   if (tid < 64) {
     sdata[tid] += sdata[tid + 64];
   }
   __syncthreads();
 }
 if (tid < 32)
   warpReduce<blockSize>(sdata, tid);
 if (tid == 0)
   atomicAdd(g_odata, sdata[0]);
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
    cudaMalloc(&gpuData, n*sizeof(float));
    cudaMalloc(&gpuRes, sizeof(float));
    double res = 0.f;
    for (int i = 0; i < n; i++)
    {
        data[i] = i;
    }

    cudaMemcpy(gpuData, data, n*sizeof(float), cudaMemcpyHostToDevice);
    
    int numBlocks = n/TILE+1;
    
    cudaMemset(gpuRes, 0, sizeof(float));
    bigReduce<<<n/(4*TILE)+1, TILE>>>(gpuRes, gpuData, n);
    cudaMemcpy(&gpuResultOnCpu, gpuRes, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "BigReduce: " << gpuResultOnCpu << std::endl;

    cudaMemset(gpuRes, 0, sizeof(float));
    bigReduceBlockLoad<<<n/(4*TILE)+1, TILE>>>(gpuRes, gpuData, n);
    cudaMemcpy(&gpuResultOnCpu, gpuRes, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "bigReduceBlockLoad: " << gpuResultOnCpu << std::endl;

    cudaMemset(gpuRes, 0, sizeof(float));
    reduce6<TILE><<<numBlocks, TILE, TILE * sizeof(int)>>>(gpuRes, gpuData, n);
    cudaMemcpy(&gpuResultOnCpu, gpuRes, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "reduce6: " << gpuResultOnCpu << std::endl;

    for (int i = 0; i < n; i++)
    {
        res += data[i];
    }
    std::cout << "Cpu: " << res << std::endl;
    return 0;
}