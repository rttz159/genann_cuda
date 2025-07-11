#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
#include "genann.h"

#define TILE_WIDTH 32
#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

__device__ double genann_act_linear_cuda(double a){
    return a;
}

__device__ double genann_act_threshold_cuda(double a){
    return a > 0;
}

__device__ double genann_act_sigmoid_cuda(double a)
{
    if (a < -45.0)
        return 0;
    if (a > 45.0)
        return 1;
    return 1.0 / (1 + exp(-a));
}

__global__ void tiled_mat_mul_kernel(double* A, double* B, double* C, int N1, int N2, int N3){
    // Ensure that TILE_WIDTH = BLOCK_SIZE
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);
    
    // Details regarding this thread
    int by = blockIdx.y;
    int bx = blockIdx.x; 

    int ty = threadIdx.y;
    int tx = threadIdx.x; 

    // Working on C[i,j]
    int i = TILE_WIDTH*by + ty;
    int j = TILE_WIDTH*bx + tx;

    // Allocating shared memory
    __shared__ double sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sh_B[TILE_WIDTH][TILE_WIDTH];

    // Parallel mat mul
    double value = 0;
    for (int phase = 0; phase < ceil((double)N2/TILE_WIDTH); phase++){
        // Load Tiles into shared memory
        if ((i < N1) && ((phase*TILE_WIDTH+tx) < N2))
          sh_A[ty][tx] = A[(i)*N2 + phase*TILE_WIDTH+tx];
        else
          sh_A[ty][tx] = 0.0f;

        if (((phase*TILE_WIDTH + ty) < N2) && (j < N3))
          sh_B[ty][tx] = B[(phase*TILE_WIDTH + ty)*N3+j];
        else
          sh_B[ty][tx] = 0.0f;
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }
    // Assigning calculated value
    if ((i < N1) && (j < N3))
      C[i*N3+j] = value;
}

// Applying activation functions
__global__ void apply_activation_kernel(double* data, int size, int activation_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        switch(activation_type){
            case 0:
                data[idx] = genann_act_linear_cuda(data[idx]);
                break;
            case 1:
                data[idx] = genann_act_threshold_cuda(data[idx]);
                break;
            case 2:
                data[idx] = genann_act_sigmoid_cuda(data[idx]);
                break;
        }
    }
}

void tiled_mat_mul_gpu(double* A, double* B, double* C, int N1, int N2, int N3){

    // Device array pointers
    double* d_A;
    double* d_B;
    double* d_C;

    // Device memory allocation
    cudaError_t err_A = cudaMalloc((void**) &d_A, N1*N2*sizeof(double));
    CUDA_CHECK(err_A);

    cudaError_t err_B = cudaMalloc((void**) &d_B, N2*N3*sizeof(double));
    CUDA_CHECK(err_B);

    cudaError_t err_C = cudaMalloc((void**) &d_C, N1*N3*sizeof(double));
    CUDA_CHECK(err_C);

    // Copying A and B to device memory
    cudaError_t err_A_ = cudaMemcpy(d_A, A, N1*N2*sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err_A_);

    cudaError_t err_B_ = cudaMemcpy(d_B, B, N2*N3*sizeof(double), cudaMemcpyHostToDevice);
    CUDA_CHECK(err_B_);

    // Kernel execution
    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dim_grid(ceil(N3/(double)(TILE_WIDTH)), ceil(N1/(double)(TILE_WIDTH)), 1);
    tiled_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N1, N2, N3);

    // Copy back results
    cudaError_t err_C_ = cudaMemcpy(C, d_C, N1*N3*sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err_C_);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void launch_activation_kernel(double* d_data, int size, int activation_function, double steepness) {
    unsigned int block_size = 256;
    unsigned int grid_size = (size + block_size - 1) / block_size;
    apply_activation_kernel<<<grid_size, block_size>>>(d_data, size, activation_function, steepness);
    CUDA_CHECK(cudaDeviceSynchronize());
}

double *genann_run_cuda(genann *ann, double *input){

    if (!ann || !input) {
        fprintf(stderr, "Error: ann or input is NULL in genann_run_cuda.\n");
        return NULL;
    }

    // Allocate device memory for input and output of each layer
    double* output; // memory layout: [inputs,outputs]
    double* weights;

    // Get information
    unsigned int num_total_neurons = ann->total_neurons;
    unsigned int num_total_weight = ann->total_weights;
    unsigned int num_input = ann->inputs;
    unsigned int num_hidden = ann->hidden;
    unsigned int num_output = ann->outputs;
    unsigned int num_hidden_layers = ann->hidden_layers;

    // Allocate initial input and output for each neuron on the device
    CUDA_CHECK(cudaMalloc((void**)&output, num_total_neurons * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(output, input, num_input * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&weights, num_total_weight * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(weights, ann->weight, num_total_weight * sizeof(double), cudaMemcpyHostToDevice));

    // Loop through each layer
    for (unsigned int l = 0; l < num_hidden_layers + 1; l++) {
        unsigned int M = ann->weight_matrices_size[l * 2]; 
        unsigned int K = ann->weight_matrices_size[l * 2 + 1];

        // Allocate device memory for the weight matrix of the current layer
        double* d_weight_matrix;
        CUDA_CHECK(cudaMalloc((void**)&d_weight_matrix, M * K * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_weight_matrix, ann->weight_matrices[l], M * K * sizeof(double), cudaMemcpyHostToDevice));

        // Allocate device memory for the output of this layer (before activation), input for the next matrix multiplication
        unsigned int next_input_size = M; // Number of neurons in the next layer
        CUDA_CHECK(cudaMalloc((void**)&d_next_layer_output, next_input_size * sizeof(double)));

        // input (1 x K), weight_matrix (M x K).
        // weight matrix as K rows, M columns (transposed).
        tiled_mat_mul_gpu(d_weight_matrix, d_current_input, d_next_layer_output, M, K, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Apply activation function
        enum fann_activationfunc_enum activation_func = ann->activations[l];
        double steepness = ann->activation_steepnesses[l];
        launch_activation_kernel(d_next_layer_output, next_input_size, activation_func, steepness);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Free the previous layer's input 
        if (l > 0) {
            CUDA_CHECK(cudaFree(d_current_input));
        }

        // The output of this layer becomes the input for the next layer
        d_current_input = d_next_layer_output;

        // Free the weight matrix for the current layer
        CUDA_CHECK(cudaFree(d_weight_matrix));
    }

    // Allocate host memory for the final output
    double* host_output = (double*)malloc(num_output_neurons * sizeof(double));
    if (!host_output) {
        fprintf(stderr, "Error: Failed to allocate host memory for final output.\n");
        CUDA_CHECK(cudaFree(d_current_input));
        return NULL;
    }

    // Copy the final output from device to host
    CUDA_CHECK(cudaMemcpy(host_output, d_current_input, num_output_neurons * sizeof(double), cudaMemcpyDeviceToHost));

    // Store final output in ann
    if (ann->final_output) {
        free(ann->final_output);
    }
    ann->final_output = host_output;

    CUDA_CHECK(cudaFree(d_current_input));

    return ann->final_output;
}