#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


const int N = 10;           
const int threadsPerBlock = 4;
const int blocksPerGrid = 4;


__global__ void sum_exp_kernel(const float* outputs, float* logits, int n) 
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int cacheIndex = threadIdx.x;

    float temp_sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        temp_sum += expf(outputs[i]);
    }

    cache[cacheIndex] = temp_sum;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        atomicAdd(&logits[0], cache[0]);
    }
}


__global__ void softmax_kernel(const float* outputs, float* logits, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float sum_val = logits[0];  

    for (int i = tid; i < n; i += stride) {
        logits[i + 1] = expf(outputs[i]) / sum_val; 
    }
}


std::vector<torch::Tensor> my_softmax_forward(torch::Tensor input)
{
    TORCH_CHECK(input.type().is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.numel() == N, "input must have N elements");

    auto logits = torch::zeros({N + 1}, torch::dtype(torch::kFloat32).device(input.device()));

    float* input_ptr = (float*) input.data_ptr<float>();
    float* logits_ptr = (float*) logits.data_ptr<float>();

    sum_exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_ptr, logits_ptr, N);

    cudaDeviceSynchronize();

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_ptr, logits_ptr, N);

    cudaDeviceSynchronize();

    return { logits };
}

torch::Tensor softmax(torch::Tensor input) {
    auto out = my_softmax_forward(input)[0];  // берём тот же метод
    return out.slice(/*dim=*/0, /*start=*/1, /*end=*/N+1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_softmax_forward", &my_softmax_forward, "my softmax forward (CUDA)");
    m.def("softmax",    &softmax,    "softmax (CUDA)");
}
