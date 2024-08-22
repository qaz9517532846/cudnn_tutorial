#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define checkCudaErrors(status) { if(status != 0) { std::cerr << "CUDA Error: " << status << std::endl; exit(1); } }
#define checkCUDNN(status) { if(status != CUDNN_STATUS_SUCCESS) { std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl; exit(1); } }

int main() {
    // 1. Initialize cuDNN
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // 2. RNN network hyperparameters
    const int input_size = 10;        // Input dimension
    const int hidden_size = 20;       // Hidden layer dimension
    const int num_layers = 1;         // Number of RNN layers
    const int batch_size = 3;         // Batch size
    const int seq_length = 5;         // Sequence length

    // 3. Create dropout descriptor
    cudnnDropoutDescriptor_t dropout_desc;
    checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_desc));

    // 4. Set dropout descriptor
    float dropout = 0.0f;  // No dropout
    void* states;
    size_t state_size;
    checkCUDNN(cudnnDropoutGetStatesSize(cudnn, &state_size));
    checkCudaErrors(cudaMalloc(&states, state_size));
    checkCUDNN(cudnnSetDropoutDescriptor(dropout_desc, cudnn, dropout, states, state_size, 0));

    // 5. Create RNN descriptor
    cudnnRNNDescriptor_t rnn_desc;
    checkCUDNN(cudnnCreateRNNDescriptor(&rnn_desc));
    checkCUDNN(cudnnSetRNNDescriptor_v6(
        cudnn, rnn_desc, hidden_size, num_layers, dropout_desc,
        CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, CUDNN_RNN_TANH,
        CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));

    // 6. Create input/output tensor descriptors
    cudnnTensorDescriptor_t x_desc[seq_length];
    cudnnTensorDescriptor_t y_desc[seq_length];
    for (int i = 0; i < seq_length; ++i) {
        checkCUDNN(cudnnCreateTensorDescriptor(&x_desc[i]));
        checkCUDNN(cudnnCreateTensorDescriptor(&y_desc[i]));
        checkCUDNN(cudnnSetTensor4dDescriptor(
            x_desc[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            batch_size, input_size, 1, 1));
        checkCUDNN(cudnnSetTensor4dDescriptor(
            y_desc[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            batch_size, hidden_size, 1, 1));
    }

    // 7. Get RNN parameter size and create filter descriptor for weights
    size_t params_size;
    checkCUDNN(cudnnGetRNNParamsSize(cudnn, rnn_desc, x_desc[0], &params_size, CUDNN_DATA_FLOAT));
    std::cout << "RNN parameters size: " << params_size << " bytes" << std::endl;

    cudnnFilterDescriptor_t w_desc;
    checkCUDNN(cudnnCreateFilterDescriptor(&w_desc));
    int filter_dimA[3] = {(int)(params_size / sizeof(float)), 1, 1};
    checkCUDNN(cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, filter_dimA));

    float* d_params;
    checkCudaErrors(cudaMalloc(&d_params, params_size));
    checkCudaErrors(cudaMemset(d_params, 0, params_size));  // Initialize to 0

    // 8. Get workspace size and allocate
    size_t workspace_size;
    checkCUDNN(cudnnGetRNNWorkspaceSize(cudnn, rnn_desc, seq_length, x_desc, &workspace_size));
    std::cout << "Workspace size: " << workspace_size << " bytes" << std::endl;

    float* d_workspace;
    checkCudaErrors(cudaMalloc(&d_workspace, workspace_size));

    // 9. Allocate output, hidden state memory
    float* d_y;
    checkCudaErrors(cudaMalloc(&d_y, batch_size * seq_length * hidden_size * sizeof(float)));

    float* d_hx;
    checkCudaErrors(cudaMalloc(&d_hx, num_layers * batch_size * hidden_size * sizeof(float)));
    checkCudaErrors(cudaMemset(d_hx, 0, num_layers * batch_size * hidden_size * sizeof(float)));  // Initialize to 0

    float* d_hy;
    checkCudaErrors(cudaMalloc(&d_hy, num_layers * batch_size * hidden_size * sizeof(float)));

    // 10. Perform RNN forward inference
    checkCUDNN(cudnnRNNForwardInference(
        cudnn, rnn_desc, seq_length, x_desc, d_hx, w_desc, d_params,
        y_desc, d_y, d_hy, d_workspace, workspace_size));

    // 11. Copy and print the output
    float h_y[batch_size * seq_length * hidden_size];
    checkCudaErrors(cudaMemcpy(h_y, d_y, batch_size * seq_length * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "RNN output:" << std::endl;
    for (int i = 0; i < batch_size * seq_length * hidden_size; ++i) {
        std::cout << h_y[i] << " ";
        if ((i + 1) % hidden_size == 0) std::cout << std::endl;
    }

    // 12. Clean up resources
    checkCudaErrors(cudaFree(d_params));
    checkCudaErrors(cudaFree(d_workspace));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_hx));
    checkCudaErrors(cudaFree(d_hy));
    checkCudaErrors(cudaFree(states));

    for (int i = 0; i < seq_length; ++i) {
        checkCUDNN(cudnnDestroyTensorDescriptor(x_desc[i]));
        checkCUDNN(cudnnDestroyTensorDescriptor(y_desc[i]));
    }

    checkCUDNN(cudnnDestroyFilterDescriptor(w_desc));
    checkCUDNN(cudnnDestroyDropoutDescriptor(dropout_desc));
    checkCUDNN(cudnnDestroyRNNDescriptor(rnn_desc));
    checkCUDNN(cudnnDestroy(cudnn));

    std::cout << "RNN forward propagation completed successfully!" << std::endl;
    return 0;
}