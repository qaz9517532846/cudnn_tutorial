#include <iostream>
#include <fstream>
#include <vector>
#include <cudnn.h>
#include <cuda_runtime.h>

#define checkCudaErrors(status) { if(status != 0) { std::cerr << "CUDA Error: " << status << std::endl; exit(1); } }
#define checkCUDNN(status) { if(status != CUDNN_STATUS_SUCCESS) { std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl; exit(1); } }

// 保存权重到文件
void saveWeights(const std::string &filename, float *weights, size_t size) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error opening file for saving weights: " << filename << std::endl;
        return;
    }
    ofs.write(reinterpret_cast<char*>(weights), size * sizeof(float));
    ofs.close();
}

// 从文件加载权重
void loadWeights(const std::string &filename, float *weights, size_t size) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error opening file for loading weights: " << filename << std::endl;
        return;
    }
    ifs.read(reinterpret_cast<char*>(weights), size * sizeof(float));
    ifs.close();
}

int main(int argc, char** argv)
{
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // 1. 创建输入张量描述符
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 28, 28));

    // 2. 创建卷积核描述符
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 5, 5));

    // 3. 创建卷积描述符
    cudnnConvolutionDescriptor_t conv_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // 4. 获取输出张量大小
    int batch_size, channels, height, width;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_descriptor, input_descriptor, kernel_descriptor, &batch_size, &channels, &height, &width));

    // 5. 创建输出张量描述符
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width));

    // 6. 分配内存
    float *d_input = nullptr;
    checkCudaErrors(cudaMalloc(&d_input, 1 * 1 * 28 * 28 * sizeof(float)));

    float *d_output = nullptr;
    checkCudaErrors(cudaMalloc(&d_output, batch_size * channels * height * width * sizeof(float)));

    float *d_kernel = nullptr;
    checkCudaErrors(cudaMalloc(&d_kernel, 1 * 1 * 5 * 5 * sizeof(float)));

    // 7. 初始化卷积核
    float h_kernel[1 * 1 * 5 * 5];
    for (int i = 0; i < 1 * 1 * 5 * 5; ++i) {
        h_kernel[i] = 0.1f; // 假设卷积核的初始化
    }
    checkCudaErrors(cudaMemcpy(d_kernel, h_kernel, 1 * 1 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice));

    // 8. 卷积前向传播
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel, conv_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, output_descriptor, d_output));

    // 9. 保存卷积核权重到文件
    saveWeights("kernel_weights.bin", h_kernel, 1 * 1 * 5 * 5);

    // 10. 从文件加载卷积核权重
    float loaded_kernel[1 * 1 * 5 * 5];
    loadWeights("kernel_weights.bin", loaded_kernel, 1 * 1 * 5 * 5);

    // 11. 打印加载的权重
    std::cout << "Loaded kernel weights:\n";
    for (int i = 0; i < 1 * 1 * 5 * 5; ++i) {
        std::cout << loaded_kernel[i] << " ";
        if ((i + 1) % 5 == 0) std::cout << std::endl; // 每行打印5个元素
    }

    // 12. 释放资源
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_kernel));

    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
    checkCUDNN(cudnnDestroy(cudnn));

    std::cout << "Model saved and loaded successfully!" << std::endl;
    return 0;
}