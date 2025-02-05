#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Параметры, как в вашем коде
// (можно и не хардкодить, а передавать из Python)
const int N = 10;           
const int threadsPerBlock = 4;
const int blocksPerGrid    = 4;

// ---------------------------------------------
// sum_exp_kernel
// ---------------------------------------------
__global__ void sum_exp_kernel(const float* outputs, float* logits, int n) 
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int cacheIndex = threadIdx.x;

    // Собираем сумму экспонент
    float temp_sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        temp_sum += expf(outputs[i]);
    }

    // Записываем частичную сумму в shared memory
    cache[cacheIndex] = temp_sum;
    __syncthreads();

    // Редукция в блоке
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // atomicAdd суммируем в logits[0]
    if (cacheIndex == 0) {
        atomicAdd(&logits[0], cache[0]);
    }
}

// ---------------------------------------------
// softmax_kernel
// ---------------------------------------------
__global__ void softmax_kernel(const float* outputs, float* logits, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float sum_val = logits[0];  // Уже посчитанная сумма

    for (int i = tid; i < n; i += stride) {
        logits[i + 1] = expf(outputs[i]) / sum_val; 
    }
}

// ---------------------------------------------
// "Обёрточные" функции для PyTorch
// ---------------------------------------------

// Эта функция будет вызвана из Python
std::vector<torch::Tensor> my_softmax_forward(torch::Tensor input)
{
    // Проверяем, что это float32 и на CUDA
    TORCH_CHECK(input.type().is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.numel() == N, "input must have N elements");

    // Создаём выходной тензор logits размером (N+1),
    // где logits[0] будет содержать сумму, а с 1 до N — softmax-значения
    auto logits = torch::zeros({N + 1}, torch::dtype(torch::kFloat32).device(input.device()));

    // Указатели на GPU-память
    float* input_ptr = (float*) input.data_ptr<float>();
    float* logits_ptr = (float*) logits.data_ptr<float>();

    // 1) Запускаем sum_exp_kernel
    //    — инициализируем logits[0] как 0 перед ядром
    sum_exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_ptr, logits_ptr, N);

    // Синхронизируемся, чтобы убедиться, что сумма записана
    cudaDeviceSynchronize();

    // 2) Запускаем softmax_kernel
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_ptr, logits_ptr, N);

    // Синхронизируем, чтобы дождаться вычислений
    cudaDeviceSynchronize();

    // Возвращаем (1) сумму [logits[0]] и (2) softmax-значения logits[1..N]
    // Но чаще вы захотите вернуть *только* softmax, например
    return { logits };
}

// Для удобства сделаем метод, который возвращает только сами softmax-значения (без суммы):
torch::Tensor softmax(torch::Tensor input) {
    auto out = my_softmax_forward(input)[0];  // берём тот же метод
    // out: size N+1, out[0] – сумма, out[1..N] – softmax
    // нам нужны только N значений, начиная с индекса 1
    // Срез (N элементов, начиная с 1):
    return out.slice(/*dim=*/0, /*start=*/1, /*end=*/N+1);
}

// Регистрация функций в PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_softmax_forward", &my_softmax_forward, "my softmax forward (CUDA)");
    m.def("softmax",    &softmax,    "softmax (CUDA)");
}