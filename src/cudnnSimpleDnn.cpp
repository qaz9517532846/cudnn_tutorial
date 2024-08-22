#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define checkCudaErrors(status) { if(status != 0) { std::cerr << "CUDA Error: " << status << std::endl; exit(1); } }
#define checkCUDNN(status) { if(status != CUDNN_STATUS_SUCCESS) { std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl; exit(1); } }

int main(int argc, char** argv)
{
    // 1. 初始化cuDNN
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // 2. 创建输入张量描述符
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 28, 28));

    // 3. 创建卷积核描述符
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 5, 5));

    // 4. 创建卷积描述符
    cudnnConvolutionDescriptor_t conv_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // 5. 获取输出张量大小
    int batch_size, channels, height, width;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_descriptor, input_descriptor, kernel_descriptor, &batch_size, &channels, &height, &width));

    // 6. 创建输出张量描述符
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width));

    // 7. 分配内存
    float* d_input = nullptr;
    checkCudaErrors(cudaMalloc(&d_input, 1 * 1 * 28 * 28 * sizeof(float)));

    float* d_output = nullptr;
    checkCudaErrors(cudaMalloc(&d_output, batch_size * channels * height * width * sizeof(float)));

    float* d_kernel = nullptr;
    checkCudaErrors(cudaMalloc(&d_kernel, 1 * 1 * 5 * 5 * sizeof(float)));

    // 初始化输入数据
    float h_input[1 * 1 * 28 * 28];
    for (int i = 0; i < 1 * 1 * 28 * 28; ++i) {
        h_input[i] = 1.0f;
    }
    checkCudaErrors(cudaMemcpy(d_input, h_input, 1 * 1 * 28 * 28 * sizeof(float), cudaMemcpyHostToDevice));

    // 初始化卷积核数据
    float h_kernel[1 * 1 * 5 * 5];
    for (int i = 0; i < 1 * 1 * 5 * 5; ++i) {
        h_kernel[i] = 1.0f;
    }
    checkCudaErrors(cudaMemcpy(d_kernel, h_kernel, 1 * 1 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice));

    // 8. 卷积前向传播
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel, conv_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, output_descriptor, d_output));

    // 9. 创建激活层描述符 (ReLU)
    cudnnActivationDescriptor_t activation_descriptor;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

    // 10. 激活前向传播
    checkCUDNN(cudnnActivationForward(cudnn, activation_descriptor, &alpha, output_descriptor, d_output, &beta, output_descriptor, d_output));

    // 提取并打印输出数据
    float h_output[batch_size * channels * height * width];
    checkCudaErrors(cudaMemcpy(h_output, d_output, batch_size * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Output:\n";
    for (int i = 0; i < batch_size * channels * height * width; ++i) {
        std::cout << h_output[i] << " ";
        if ((i + 1) % width == 0) std::cout << std::endl;
    }

    // 11. 释放资源
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_kernel));

    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
    checkCUDNN(cudnnDestroyActivationDescriptor(activation_descriptor));
    checkCUDNN(cudnnDestroy(cudnn));

    std::cout << "DNN Forward Propagation Completed Successfully!" << std::endl;
    return 0;
}