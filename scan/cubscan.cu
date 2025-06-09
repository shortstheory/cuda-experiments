#include <cub/cub.cuh>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    int N=1000;

   if (argc > 1)
   {
       N = std::stoi(argv[1]);
   }

    std::vector<float> h_input(N), h_output(N);

    // Fill input data (e.g., with 1s or i+1)
    for (int i = 0; i < N; ++i) h_input[i] = i;

    float* d_input;
    float* d_output;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // First call to determine temp storage size
    cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes,
        d_input, d_output, N);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Second call to actually run the scan
    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_input, d_output, N);

    // Copy output back to host
    cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output
    // std::cout << "Scan result:\n";
    // for (int i = 0; i < N; ++i) {
    //     std::cout << h_output[i] << " ";
    // }
    // std::cout << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_storage);

    return 0;
}