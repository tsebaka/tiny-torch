#include <cuda_runtime.h>

__global__ void relu_kernel(float* a, size_t n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n; i += stride)
        a[i] = (a[i] > 0.0f) ? a[i] : 0.0f;
}

void relu_cuda(float* input, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, n);
    cudaDeviceSynchronize();
}
