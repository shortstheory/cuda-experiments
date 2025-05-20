#include <math.h>


#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>
#include <cstdlib>
#include <ctime>
#include <cooperative_groups.h>
#include <iostream>
#include <string>  // Required for std::stoi()
constexpr int TILE = 128;


__global__ void reduce0(int *g_odata, int *g_idata, int n) {
 extern __shared__ int sdata[];
 // each thread loads one element from global to shared mem
 unsigned int tid = threadIdx.x;
 unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i < n) {
   sdata[tid] = g_idata[i];
 } else {
   sdata[tid] = 0;
 }
 __syncthreads();
 // do reduction in shared mem
 for (unsigned int s = 1; s < blockDim.x; s *= 2) {
   if (tid % (2 * s) == 0) {
     sdata[tid] += sdata[tid + s];
   }
   __syncthreads();
 }
 // write result for this block to global mem
 if (tid == 0)
   g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduce1(int *g_odata, int *g_idata, int n) {
 extern __shared__ int sdata[];
 // each thread loads one element from global to shared mem
 unsigned int tid = threadIdx.x;
 unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;


 if (i < n) {
   sdata[tid] = g_idata[i];
 } else {
   sdata[tid] = 0;
 }
 __syncthreads();
 // do reduction in shared mem
 for (unsigned int s = 1; s < blockDim.x; s *= 2) {
   int index = 2 * s * tid;
   if (index < blockDim.x) {
     sdata[index] += sdata[index + s];
   }
   __syncthreads();
 } // write result for this block to global mem
 if (tid == 0)
   g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduce2(int *g_odata, int *g_idata, int n) {
 extern __shared__ int sdata[];
 // each thread loads one element from global to shared mem
 unsigned int tid = threadIdx.x;
 unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;


 if (i < n) {
   sdata[tid] = g_idata[i];
 } else {
   sdata[tid] = 0;
 }
 __syncthreads();
 // do reduction in shared mem
 for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
   if (tid < s) {
     sdata[tid] += sdata[tid + s];
   }
   __syncthreads();
 }
 if (tid == 0)
   g_odata[blockIdx.x] = sdata[0];
}


template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
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
__global__ void reduce6(int *g_odata, int *g_idata, unsigned int n) {
 extern __shared__ int sdata[];
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
   g_odata[blockIdx.x] = sdata[0];
}

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

int main(int argc, char **argv)
{
  
   int n = 1000*1000;
   if (argc > 1)
   {
       n = std::stoi(argv[1]);
   }
   std::cout << "N: " << n << std::endl;
   int* data = new int[n];
   int* gpuData;
   int* gpuRes;
   int numBlocks = n / TILE + 1;


   cudaMalloc(&gpuData, n*sizeof(int));
   cudaMalloc(&gpuRes, n*sizeof(int));


   for (int i = 0; i < n; i++)
   {
       data[i] = (rand() % 100) + 1;;
   }


   cudaMemcpy(gpuData, data, n*sizeof(int), cudaMemcpyHostToDevice);


   reduce0<<<numBlocks,TILE, TILE*sizeof(int)>>>(gpuRes, gpuData, n);
   reduce1<<<numBlocks, TILE, TILE * sizeof(int)>>>(gpuRes, gpuData, n);
   reduce2<<<numBlocks, TILE, TILE * sizeof(int)>>>(gpuRes, gpuData, n);
   reduce6<TILE><<<numBlocks, TILE, TILE * sizeof(int)>>>(gpuRes, gpuData, n);


   return 0;
}

