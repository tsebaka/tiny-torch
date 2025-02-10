#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


const int threadsPerBlock = 256;


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

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
    }

    if (cacheIndex == 0) {
        atomicAdd(&logits[0], cache[0]); 
    }
}


__global__ void softmax_kernel(const float* outputs, float* logits, float* softmax_out, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float sum_val = logits[0];  

    for (int i = tid; i < n; i += stride) {
        softmax_out[i] = expf(outputs[i]) / sum_val; 
    }
}


std::vector<torch::Tensor> my_softmax_forward(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    int n = input.numel();  
    auto logits = torch::zeros({1}, torch::dtype(torch::kFloat32).device(input.device())); 
    auto softmax_out = torch::zeros({n}, torch::dtype(torch::kFloat32).device(input.device()));

    float* input_ptr = input.data_ptr<float>();
    float* logits_ptr = logits.data_ptr<float>();
    float* softmax_out_ptr = softmax_out.data_ptr<float>();

    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    sum_exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_ptr, logits_ptr, n);
    cudaDeviceSynchronize();

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_ptr, logits_ptr, softmax_out_ptr, n);
    cudaDeviceSynchronize();

    return { softmax_out };
}

torch::Tensor softmax(torch::Tensor input) {
    return my_softmax_forward(input)[0];  
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_softmax_forward", &my_softmax_forward, "my softmax forward (CUDA)");
    m.def("softmax", &softmax, "softmax (CUDA)");
}
