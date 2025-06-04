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
constexpr int BLOCK_SIZE = PARTITION_SIZE*4;


enum BlockState {
   BLOCK_WAIT=0,
   BLOCK_SUM_READY=1,
   BLOCK_SUM_DONE=2
};


__global__ void myShuffleScan(float *data, int* blockCounter, BlockState* blockStates, float* blockResults, int n) {
   __shared__ int sBlockNum;
   __shared__ int sPrefixSumIndex;
   if (threadIdx.x == 0)
   {
       sBlockNum = atomicAdd(blockCounter, 1);
       sPrefixSumIndex = max(sBlockNum-1,0);
   }
   __syncthreads();
   int offset = blockIdx.x * blockDim.x;
   int offset1 = sBlockNum * blockDim.x;

   if (threadIdx.x == 0)
   {
        printf("Offset %d offset1 %d\n",offset,offset1);
   }
   data = data + offset1;


   __shared__ float sharedData[BLOCK_SIZE/PARTITION_SIZE];


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


   if (idx >= PARTITION_SIZE)
   {
       threadValue += sharedData[idx/PARTITION_SIZE-1];
   }

/*
   if ((idx+1) % BLOCK_SIZE == 0 || idx + 1 == n)
   {
       if (sBlockNum > 0)
       {
         blockStates[sBlockNum] = BLOCK_SUM_READY;
         blockResults[sBlockNum] = threadValue;
       }
       else
       {
           blockStates[0] = BLOCK_SUM_DONE;
           blockResults[0] = threadValue;
       }
         printf("Index %d Val %f\n", sBlockNum, blockResults[sBlockNum]);
   }


   __shared__ float sPrefixSum;
   if (threadIdx.x == 0)
   {
     sPrefixSum = 0.f;
     while (blockStates[sPrefixSumIndex] != BLOCK_SUM_DONE) {
       if (blockStates[sPrefixSumIndex] == BLOCK_SUM_READY) {
         sPrefixSum += blockResults[sPrefixSumIndex];
       }
       sPrefixSumIndex--;
     }
     sPrefixSum += blockResults[sPrefixSumIndex];
     blockStates[sBlockNum] = BLOCK_SUM_DONE;
     printf("Index %d Val %f\n", sBlockNum, sPrefixSum);
   }
*/

   if (idx < n)
   {
       data[idx] = threadValue;//+sPrefixSum;
   }
}




int main(int argc, char **argv)
{
   int n = 128;


   if (argc > 1)
   {
       n = std::stoi(argv[1]);
   }
   int numBlocks = (n + BLOCK_SIZE -1) / BLOCK_SIZE;
   std::cout << "N: " << n << std::endl;
   float* data = new float[n];
   float* gpuResultCpu = new float[n];
   float* gpuData;
   int* blockCounterGpu;
   float* blockResultGpu;
   BlockState* blockStatesGpu;
   cudaError_t error = cudaMalloc(&gpuData, n*sizeof(float));
    cudaMalloc(&blockStatesGpu, numBlocks * sizeof(BlockState));
    cudaMalloc(&blockResultGpu, numBlocks * sizeof(float));
    cudaMalloc(&blockCounterGpu, sizeof(int));
   cudaMemset(blockStatesGpu, 0, numBlocks * sizeof(BlockState));
   cudaMemset(blockResultGpu, 0, numBlocks * sizeof(float));
   cudaMemset(blockCounterGpu, 0, sizeof(int));


   std::cout << cudaGetErrorString(error) << std::endl;
   for (int i = 0; i < n; i++)
   {
       data[i] = i;
   }


   cudaMemcpy(gpuData, data, n*sizeof(float), cudaMemcpyHostToDevice);
   myShuffleScan<<<numBlocks, BLOCK_SIZE>>>(gpuData, blockCounterGpu, blockStatesGpu,
                                      blockResultGpu, n);


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

