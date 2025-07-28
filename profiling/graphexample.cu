#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
// #include <nvtx3/nvtx3.hpp>
#define N 500000 // tuned such that kernel takes a few microseconds

__global__ void shortKernel(float * out_d, float * in_d){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<N) out_d[idx]=1.23*in_d[idx];
}

int main()
{
    cudaStream_t s_1;
    cudaStreamCreate(&s_1);
#define NSTEP 1000
#define NKERNEL 20
int threads = 512;
int blocks = 100;
    size_t size = N * sizeof(float);
    float* d_data;

    cudaMalloc(&d_data, size);

// start CPU wallclock timer
// for(int istep=0; istep<NSTEP; istep++){
//   for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
//     shortKernel<<<blocks, threads, 0, s_1>>>(d_data, d_data);
//     cudaStreamSynchronize(s_1);
//   }
// }
for(int istep=0; istep<NSTEP; istep++){
  for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
    shortKernel<<<blocks, threads, 0, s_1>>>(d_data, d_data);
  }
  cudaStreamSynchronize(s_1);
}

// bool graphCreated=false;
// cudaGraph_t graph;
// cudaGraphExec_t instance;
// for(int istep=0; istep<NSTEP; istep++){
//   if(!graphCreated){
//     cudaStreamBeginCapture(s_1, cudaStreamCaptureModeGlobal);
//     for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
//       shortKernel<<<blocks, threads, 0, s_1>>>(d_data, d_data);
//     }
//     cudaStreamEndCapture(s_1, &graph);
//     cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
//     graphCreated=true;
//   }
//   cudaGraphLaunch(instance, s_1);
//   cudaStreamSynchronize(s_1);
// }
    return 0;
}