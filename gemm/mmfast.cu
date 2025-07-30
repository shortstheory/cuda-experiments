1	/**									
2	 * matrix_mul_4096.cu									
3	 *									
4	 * nvcc -O3 -std=c++17 -o matmul_4096 matrix_mul_4096.cu									
5	 *									
6	 * Launches the tiled batched-GEMM kernel supplied by the user on									
7	 * A (1 × 4096 × 4096)  *  B (1 × 4096 × 4096)  →  C (1 × 4096 × 4096)									
8	 *									
9	 * ────────────────────────────────────────────────────────────────────									
10	 * The code shows only the minimum boiler-plate needed:									
11	 *   • allocate & fill host data									
12	 *   • copy to GPU									
13	 *   • prepare shape / stride metadata									
14	 *   • launch the kernel									
15	 *   • copy result back & sanity-check against cuBLAS (optional)									
16	 */									
17										
18	#include <cuda_runtime.h>									
19	#include <cstdlib>									
20	#include <cstdio>									
21	#include <cassert>									
22	#include <cmath>									
23										
24	#define TILE 16               // same tile size the kernel expects									
25	#define CHECK(call) do {                                          \									
26	  cudaError_t err = (call);                                       \									
27	  if (err != cudaSuccess) {                                       \									
28	    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,    \									
29	            cudaGetErrorString(err));                             \									
30	    std::exit(EXIT_FAILURE);                                      \									
31	  }                                                               \									
32	} while(0)									
33										
34	// ───────────────────────────────── device helpers ────────────────									
35	__device__ __forceinline__									
36	int index_to_position(const int idx[3], const int strides[3], int dims) {									
37	  int pos = 0;									
38	  #pragma unroll									
39	  for (int d = 0; d < dims; ++d) pos += idx[d] * strides[d];
40	  return pos;									
41	}									
42										
43	// ──────────────────────────────── user kernel ────────────────────									
44	__global__ void MatrixMultiplyKernel(	7	63	0.02%	32					
45	  float* out,									
46	  const int* out_shape,									
47	  const int* out_strides,									
48	  const float* a_storage,									
49	  const int* a_shape,									
50	  const int* a_strides,									
51	  const float* b_storage,									
52	  const int* b_shape,									
53	  const int* b_strides									
54	) {									
55	/**									
56	 * Multiply two (compact) matrices into an output (also comapct) matrix. Matrix a and b are both in a batch									
57	 * format, with shape [batch_size, m, n], [batch_size, n, p].									
58	 * Requirements:									
59	 * - All data must be first moved to shared memory.									
60	 * - Only read each cell in a and b once.									
61	 * - Only write to global memory once per kernel.									
62	 * There is guarantee that a_shape[0] == b_shape[0], a_shape[2] == b_shape[1],									
63	 * and out_shape[0] == a_shape[0], out_shape[1] == b_shape[1]									
64	 *									
65	 * Args:									
66	 *   out: compact 1D array of size batch_size x m x p to write the output to									
67	 *   out_shape: shape of the output array									
68	 *   out_strides: strides of the output array									
69	 *   a_storage: compact 1D array of size batch_size x m x n									
70	 *   a_shape: shape of the a array									
71	 *   a_strides: strides of the a array									
72	 *   b_storage: comapct 2D array of size batch_size x n x p									
73	 *   b_shape: shape of the b array									
74	 *   b_strides: strides of the b array									
75	 *									
76	 * Returns:									
77	 *   None (Fills in out array)									
78	 */									
79										
80	  __shared__ float a_shared[TILE+1][TILE+1];									
81	  __shared__ float b_shared[TILE+1][TILE+1];									
82										
83	  // In each block, we will compute a batch of the output matrix									
84	  // All the threads in the block will work together to compute this batch									
85	  int batch = blockIdx.z;	7	15	< 0.01%	32					
86	  int a_batch_stride = a_shape[0] > 1 ? a_strides[0] : 0;									
87	  int b_batch_stride = b_shape[0] > 1 ? b_strides[0] : 0;									
88										
89										
90	  /// BEGIN ASSIGN1_2									
91	  /// TODO									
92	  // Hints:									
93	  // 1. Compute the row and column of the output matrix this block will compute									
94	  // 2. Compute the position in the output array that this thread will write to									
95	  // 3. Iterate over tiles of the two input matrices, read the data into shared memory									
96	  // 4. Synchronize to make sure the data is available to all threads									
97	  // 5. Compute the output tile for this thread block									
98	  // 6. Synchronize to make sure all threads are done computing the output tile for (row, col)									
99	  // 7. Write the output to global memory									
100										
101	  int i = blockIdx.x;
102	  int j = blockIdx.y;
103	  int aIndex[3];									
104	  int bIndex[3];									
105	  int cIndex[3];									
106	  aIndex[0] = batch;									
107	  bIndex[0] = batch;									
108	  cIndex[0] = batch;									
109	  cIndex[1] = i*blockDim.x+threadIdx.x;	13	509	0.01%	32					
110	  cIndex[2] = j*blockDim.y+threadIdx.y;	14	1.02K	0.01%	32					
111	  float accum = 0.f;									
112										
113	  for (int k = 0; k < a_shape[2]; k+=TILE)	19	720K	5.31%	31.9	Global(3)	Load(3)	32(3)		
114	  {									
115	      aIndex[1] = cIndex[1];									
116	      aIndex[2] = k+threadIdx.y;	21	1.03K	1.31%	32					
117										
118	      bIndex[1] = k+threadIdx.x;	23	2.98K	1.31%	32					
119	      bIndex[2] = cIndex[2];									
120	      // printf("A Index %d,%d B Index %d,%d\n",aIndex[1],aIndex[2],bIndex[1],bIndex[2]);									
121										
122	      if (aIndex[1] < a_shape[1] && aIndex[2] < a_shape[2])	23	114K	7.88%	32	Global(2)	Load(2)	32(2)		
123	      {									
124	          int linearAIndex = index_to_position(aIndex, a_strides, 3);	15	7.07K	0.02%	32	Global(3)	Load(3)	32(3)		
125	          a_shared[threadIdx.x][threadIdx.y] = a_storage[linearAIndex];	23	159K	3.93%	32	Global(2)	Load(2)	32(2)		49.90%
126	      } else {									
127	          a_shared[threadIdx.x][threadIdx.y] = 0.f;									
128	      }									
129	      if (bIndex[1] < b_shape[1] && bIndex[2] < b_shape[2])	21	391K	7.91%	32	Global(4)	Load(4)	32(4)		
130	      {									
131	          int linearBIndex = index_to_position(bIndex, b_strides, 3);	22	9.76K	0.04%	32	Global(3)	Load(3)	32(3)		
132	          b_shared[threadIdx.y][threadIdx.x] = b_storage[linearBIndex];	20	843K	7.86%	32	Global(2), Shared(4)	Load(2), Store(4)	32(6)	100.00%	49.90%
133	      } else{									
134	          b_shared[threadIdx.y][threadIdx.x] = 0.f;									
135	      }									
136	      __syncthreads();	16	82.2K	1.31%	32					
137										
138	      // float4* a_vec = reinterpret_cast<float4*>(a_shared[threadIdx.x]);									
139	      #pragma unroll									
140	      for (int tileIdx = 0; tileIdx < TILE/4; tileIdx++)									
141	      {									
142	        // float4 a_val = a_vec[tileIdx];									
143	        accum += a_shared[threadIdx.x][tileIdx]*b_shared[threadIdx.y][tileIdx] + a_shared[threadIdx.x][tileIdx+1]*b_shared[threadIdx.y][tileIdx+1] +a_shared[threadIdx.x][tileIdx+2]*b_shared[threadIdx.y][tileIdx+2] + a_shared[threadIdx.x][tileIdx+3]*b_shared[threadIdx.y][tileIdx+3];	32	724K	44.58%	32	Shared(28)	Load(28)	32(28)		
144	      }									
145	      // c_shared[threadIdx.x][threadIdx.y] = accum;									
146	  }									
147	  if (cIndex[1] < out_shape[1] && cIndex[2] < out_shape[2])	8	1.65K	0.05%	32	Global(2)	Load(2)	32(2)		
148	  {									
149	      int linearOutIndex = index_to_position(cIndex, out_strides, 3);	10	1.58K	0.04%	32	Global(3)	Load(3)	32(3)		
150	      out[linearOutIndex] = accum;// c_shared[threadIdx.x][threadIdx.y];	5	81	0.01%	32	Global	Store	32		0.19%
151	  }									
152	  /// END ASSIGN1_2									
153	}	1	729	< 0.01%	32					
154										
155	// ──────────────────────────────── host main ───────────────────────									
156	int main() {									
157	  constexpr int B = 1, M = 4096, N = 4096, P = 4096;									
158	  const size_t bytesA = size_t(B) * M * N * sizeof(float);									
159	  const size_t bytesB = size_t(B) * N * P * sizeof(float);									
160	  const size_t bytesC = size_t(B) * M * P * sizeof(float);									
161										
162	  // Allocate pinned host buffers									
163	  float *hA, *hB, *hC;									
164	  CHECK(cudaMallocHost(&hA, bytesA));									
165	  CHECK(cudaMallocHost(&hB, bytesB));									
166	  CHECK(cudaMallocHost(&hC, bytesC));									
167										
168	  // Fill with some deterministic data									
169	  for (size_t i = 0; i < (bytesA/sizeof(float)); ++i) hA[i] = 1.f;									
170	  for (size_t i = 0; i < (bytesB/sizeof(float)); ++i) hB[i] = 1.f;									
171										
172	  // Allocate device buffers									
173	  float *dA, *dB, *dC;									
174	  CHECK(cudaMalloc(&dA, bytesA));									
175	  CHECK(cudaMalloc(&dB, bytesB));									
176	  CHECK(cudaMalloc(&dC, bytesC));									
177										
178	  CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));									
179	  CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));									
180										
181	  // shape & stride metadata ---------------------------------------									
182	  int h_a_shape[3]    = {B, M, N};									
183	  int h_b_shape[3]    = {B, N, P};									
184	  int h_out_shape[3]  = {B, M, P};									
185										
186	  int h_a_strides[3]  = {M * N, N, 1};									
187	  int h_b_strides[3]  = {N * P, P, 1};									
188	  int h_out_strides[3]= {M * P, P, 1};									
189										
190	  int *d_a_shape, *d_b_shape, *d_out_shape;									
191	  int *d_a_strides, *d_b_strides, *d_out_strides;									
192	  CHECK(cudaMalloc(&d_a_shape,   3*sizeof(int)));									
193	  CHECK(cudaMalloc(&d_b_shape,   3*sizeof(int)));									
194	  CHECK(cudaMalloc(&d_out_shape, 3*sizeof(int)));									
195	  CHECK(cudaMalloc(&d_a_strides,   3*sizeof(int)));									
196	  CHECK(cudaMalloc(&d_b_strides,   3*sizeof(int)));									
197	  CHECK(cudaMalloc(&d_out_strides, 3*sizeof(int)));									
198										
199	  CHECK(cudaMemcpy(d_a_shape,   h_a_shape,   3*sizeof(int), cudaMemcpyHostToDevice));									
200	  CHECK(cudaMemcpy(d_b_shape,   h_b_shape,   3*sizeof(int), cudaMemcpyHostToDevice));									
201	  CHECK(cudaMemcpy(d_out_shape, h_out_shape, 3*sizeof(int), cudaMemcpyHostToDevice));									
202	  CHECK(cudaMemcpy(d_a_strides, h_a_strides, 3*sizeof(int), cudaMemcpyHostToDevice));									
203	  CHECK(cudaMemcpy(d_b_strides, h_b_strides, 3*sizeof(int), cudaMemcpyHostToDevice));									
204	  CHECK(cudaMemcpy(d_out_strides, h_out_strides, 3*sizeof(int), cudaMemcpyHostToDevice));									
205										
206	  // Kernel launch configuration -----------------------------------									
207	  dim3 block(TILE, TILE, 1);									
208	  dim3 grid((M + TILE - 1) / TILE,									
209	            (P + TILE - 1) / TILE,									
210	            B);									
211										
212	  MatrixMultiplyKernel<<<grid, block>>>(									
213	      dC, d_out_shape, d_out_strides,									
214	      dA, d_a_shape, d_a_strides,									
215	      dB, d_b_shape, d_b_strides);									
216	  CHECK(cudaGetLastError());									
217	  CHECK(cudaDeviceSynchronize());									
218										
219	  // Copy result back									
220	  CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));									
221										
222	  // (optional) quick sanity check – C should be all N when A,B filled with 1s									
223	  double max_err = 0.;									
224	  for (size_t i = 0; i < (bytesC/sizeof(float)); ++i)									
225	    max_err = fmax(max_err, fabs(hC[i] - float(N)));									
226	  printf("max error = %g (should be 0)\n", max_err);									
227										
228	  // clean up									
229	  CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));									
230	  CHECK(cudaFree(d_a_shape)); CHECK(cudaFree(d_b_shape)); CHECK(cudaFree(d_out_shape));									
231	  CHECK(cudaFree(d_a_strides)); CHECK(cudaFree(d_b_strides)); CHECK(cudaFree(d_out_strides));									
232	  CHECK(cudaFreeHost(hA)); CHECK(cudaFreeHost(hB)); CHECK(cudaFreeHost(hC));									
233	  return 0;									
234	}									
