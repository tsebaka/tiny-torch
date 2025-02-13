#include <cuda_runtime.h>

_global__ void visualizeGridThreads(matrix1, matrix2, mat_mul, N) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_col = gridDim.x * blockDim.x;
    int stride_row = gridDim.y * blockDim.y;

    for (int i = row; i < N; i += stride_row) {
        for (int j = col; j < N; i += stride_col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += matrix1[i * N + k] * matrix2[k * N + j];
            }
            mat_mul[i * N + j] = sum;
        }
    }
}
