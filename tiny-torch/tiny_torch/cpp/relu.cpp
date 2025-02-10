#include <torch/extension.h>

extern void relu_cuda(float* input, size_t n);

torch::Tensor relu(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    size_t n = input.numel();
    float* input_ptr = input.data_ptr<float>();

    relu_cuda(input_ptr, n);

    return input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu", &relu, "ReLU activation (CUDA)");
}
