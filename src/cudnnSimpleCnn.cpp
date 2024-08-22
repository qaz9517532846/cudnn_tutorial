#include <iostream>
#include <vector>
#include <cudnn.h>
#include <cuda_runtime.h>

#define checkCudaErrors(status) { if(status != 0) { std::cerr << "CUDA Error: " << status << std::endl; exit(1); } }
#define checkCUDNN(status) { if(status != CUDNN_STATUS_SUCCESS) { std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl; exit(1); } }

int main(int argc, char** argv)
{
    // 1. 初始化 cuDNN
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // 2. 创建输入张量描述符 (NCHW: batch size, channels, height, width)
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 28, 28));

    // 3. 创建卷积核描述符
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 20, 1, 5, 5));

    // 4. 创建卷积描述符 (padding, stride)
    cudnnConvolutionDescriptor_t conv_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_descriptor, 2, 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // 5. 获取卷积输出张量大小
    int batch_size, channels, height, width;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_descriptor, input_descriptor, kernel_descriptor, &batch_size, &channels, &height, &width));

    // 6. 创建卷积输出张量描述符
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width));

    // 7. 创建池化层描述符
    cudnnPoolingDescriptor_t pooling_descriptor;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    checkCUDNN(cudnnSetPooling2dDescriptor(pooling_descriptor, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2));

    // 8. 获取池化层输出张量大小
    int pooled_height, pooled_width;
    checkCUDNN(cudnnGetPooling2dForwardOutputDim(pooling_descriptor, output_descriptor, &batch_size, &channels, &pooled_height, &pooled_width));

    // 9. 创建池化层输出张量描述符
    cudnnTensorDescriptor_t pooling_output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&pooling_output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(pooling_output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, pooled_height, pooled_width));

    // 10. 分配内存
    float *d_input, *d_output, *d_pooling_output, *d_kernel;
    checkCudaErrors(cudaMalloc(&d_input, 1 * 1 * 28 * 28 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_output, batch_size * channels * height * width * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pooling_output, batch_size * channels * pooled_height * pooled_width * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_kernel, 20 * 1 * 5 * 5 * sizeof(float)));

    // 11. 初始化卷积核
    float h_kernel[20 * 1 * 5 * 5];
    for (int i = 0; i < 20 * 1 * 5 * 5; ++i) {
        h_kernel[i] = 0.1f;
    }
    checkCudaErrors(cudaMemcpy(d_kernel, h_kernel, 20 * 1 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice));

    // 12. 卷积前向传播
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel, conv_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, output_descriptor, d_output));

    // 13. 激活 (ReLU)
    cudnnActivationDescriptor_t activation_descriptor;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(cudnnActivationForward(cudnn, activation_descriptor, &alpha, output_descriptor, d_output, &beta, output_descriptor, d_output));

    // 14. 池化前向传播
    checkCUDNN(cudnnPoolingForward(cudnn, pooling_descriptor, &alpha, output_descriptor, d_output, &beta, pooling_output_descriptor, d_pooling_output));

    // 15. 输出一些池化后的数据 (演示用)
    float h_pooling_output[batch_size * channels * pooled_height * pooled_width];
    checkCudaErrors(cudaMemcpy(h_pooling_output, d_pooling_output, batch_size * channels * pooled_height * pooled_width * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Pooled output:\n";
    for (int i = 0; i < batch_size * channels * pooled_height * pooled_width; ++i) {
        std::cout << h_pooling_output[i] << " ";
        if ((i + 1) % pooled_width == 0) std::cout << std::endl;
    }

    // 16. 清理资源
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_pooling_output));
    checkCudaErrors(cudaFree(d_kernel));
    
    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
    checkCUDNN(cudnnDestroyPoolingDescriptor(pooling_descriptor));
    checkCUDNN(cudnnDestroyActivationDescriptor(activation_descriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(pooling_output_descriptor));
    checkCUDNN(cudnnDestroy(cudnn));

    std::cout << "CNN forward propagation completed successfully!" << std::endl;
    return 0;
}