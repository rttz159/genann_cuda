#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
#include "genann_cuda.cuh"
#include "genann.h"

__global__ void tiled_mat_mul_kernel(float* A, float* B, float* C,
                                     int N1, int N2, int N3,
                                     bool transpose_A, bool transpose_B)
{
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);
    
    int by = blockIdx.y;
    int bx = blockIdx.x; 

    int ty = threadIdx.y;
    int tx = threadIdx.x; 

    // Working on C[i,j]
    int i = TILE_WIDTH*by + ty;
    int j = TILE_WIDTH*bx + tx;

    // Allocating shared memory
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    // Parallel mat mul
    float value = 0;
    for (int phase = 0; phase < ((N2 + TILE_WIDTH - 1) / TILE_WIDTH); phase++){
        if (!transpose_A) {
            sh_A[ty][tx] = (i < N1 && (phase*TILE_WIDTH+tx) < N2)
                        ? A[i * N2 + phase*TILE_WIDTH + tx]
                        : 0.0f;
        } else {
            sh_A[ty][tx] = ((phase*TILE_WIDTH+tx) < N2 && i < N1)
                        ? A[(phase*TILE_WIDTH + tx)*N1 + i]
                        : 0.0f;
        }

        if (!transpose_B) {
            sh_B[ty][tx] = ((phase*TILE_WIDTH + ty) < N2 && j < N3)
                        ? B[(phase*TILE_WIDTH + ty)*N3 + j]
                        : 0.0f;
        } else {
            sh_B[ty][tx] = (j < N2 && (phase*TILE_WIDTH + ty) < N3)
                        ? B[j*N2 + (phase*TILE_WIDTH + ty)]
                        : 0.0f;
        }
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }
    if ((i < N1) && (j < N3))
      C[i*N3+j] = value;
}

// Applying activation functions
__global__ void apply_activation_kernel(float* data, int size, int activation_type) {
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

// Set bias on the array
__global__ void set_bias(float* d_input, int bias_index) {
    d_input[bias_index] = -1.0;
}

void launch_activation_kernel(float* d_data, int size, int activation_function) {
    unsigned int block_size = 256;
    unsigned int grid_size = (size + block_size - 1) / block_size;
    apply_activation_kernel<<<grid_size, block_size>>>(d_data, size, activation_function);
    CUDA_CHECK(cudaDeviceSynchronize());
}

float *genann_run_cuda(genann *ann, float const *inputs, enum GenannRunType run_type){

    if (!ann || !inputs) {
        fprintf(stderr, "Error: ann or input is NULL in genann_run_cuda.\n");
        return NULL;
    }

    unsigned int num_total_neurons = ann->total_neurons;
    unsigned int num_total_weight = ann->total_weights;
    unsigned int num_input = ann->inputs;
    unsigned int num_hidden = ann->hidden;
    unsigned int num_output = ann->outputs;
    unsigned int num_hidden_layers = ann->hidden_layers;

    enum GenannActivationType activation_hidden_type = ann->activation_hidden_type;
    enum GenannActivationType activation_output_type = ann->activation_output_type;

    // Calculate the number of bias neuron that need to be pre-appended to the output array
    unsigned int num_neuron_append = num_hidden_layers + 1;

    // Allocate host memory for output
    float* host_output = (float*)malloc((num_total_neurons + num_neuron_append) * sizeof(float));

    // Allocate initial input and output for each neuron on the device, bias also included
    if (ann->d_output == NULL) CUDA_CHECK(cudaMalloc((void**)&ann->d_output, (num_total_neurons + num_neuron_append) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(ann->d_output, inputs, num_input * sizeof(float), cudaMemcpyHostToDevice));
    
    // Set the bias neuron
    set_bias<<<1,1>>>(ann->d_output, num_input);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate the weights to the device
    if (ann->d_weights == NULL) CUDA_CHECK(cudaMalloc((void**)&ann->d_weights, num_total_weight * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(ann->d_weights, ann->weight, num_total_weight * sizeof(float), cudaMemcpyHostToDevice));
    
    // Device pointer arithemtic
    float* d_input = ann->d_output;
    float* d_result = ann->d_output + num_input + 1;
    float* d_weight = ann->d_weights;

    // Dimension initialization for block dimension
    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);

    if(!num_hidden){
        dim3 dim_grid(ceil(num_output/(float)(TILE_WIDTH)), ceil((num_input + 1)/(float)(TILE_WIDTH)), 1); // Dimension initialization for grid dimension
        // Kernel execution with no hidden layer
        tiled_mat_mul_kernel<<<dim_grid, dim_block>>>(d_input, d_weight, d_result, 1, (num_input + 1), num_output, false, false);
        CUDA_CHECK(cudaDeviceSynchronize());
        launch_activation_kernel(d_result, num_output, activation_output_type);
        CUDA_CHECK(cudaDeviceSynchronize());
    }else{
        dim3 dim_grid(ceil(num_hidden/(float)(TILE_WIDTH)), ceil((num_input + 1)/(float)(TILE_WIDTH)), 1); // Dimension initialization for grid dimension
        
        // Kernel execution for input to hidden layer
        tiled_mat_mul_kernel<<<dim_grid, dim_block>>>(d_input, d_weight, d_result, 1, (num_input + 1), num_hidden, false, false);
        CUDA_CHECK(cudaDeviceSynchronize());
        launch_activation_kernel(d_result, num_hidden, activation_hidden_type);
        CUDA_CHECK(cudaDeviceSynchronize());
        d_input += (num_input + 1);
        d_weight += ((num_input + 1) * num_hidden);
        set_bias<<<1,1>>>(d_result, (num_hidden)); // Set the bias neuron
        CUDA_CHECK(cudaDeviceSynchronize());
        d_result += (num_hidden + 1);


        // Kernel execution for subsequent hidden layer
        for(int i = 1; i < num_hidden_layers; i++){
            dim3 dim_grid_hidden(ceil(num_hidden/(float)(TILE_WIDTH)), ceil((num_hidden + 1)/(float)(TILE_WIDTH)), 1); // Dimension initialization for grid dimension
            
            tiled_mat_mul_kernel<<<dim_grid_hidden, dim_block>>>(d_input, d_weight, d_result, num_hidden, (num_hidden + 1), num_hidden, false, false);
            CUDA_CHECK(cudaDeviceSynchronize());
            launch_activation_kernel(d_result, num_hidden, activation_hidden_type);
            CUDA_CHECK(cudaDeviceSynchronize());
            d_input += (num_hidden + 1);
            d_weight += ((num_hidden + 1) * num_hidden);
            set_bias<<<1,1>>>(d_result, (num_hidden)); // Set the bias neuron
            CUDA_CHECK(cudaDeviceSynchronize());
            d_result += (num_hidden + 1);
        }

        dim3 dim_grid_output(ceil(num_output/(float)(TILE_WIDTH)), ceil((num_hidden + 1)/(float)(TILE_WIDTH)), 1); // Dimension initialization for grid dimension

        // Kernel execution for hidden to output layer
        tiled_mat_mul_kernel<<<dim_grid_output, dim_block>>>(d_input, d_weight, d_result, 1, (num_hidden + 1), num_output, false, false);
        CUDA_CHECK(cudaDeviceSynchronize());
        launch_activation_kernel(d_result, num_output, activation_output_type);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(host_output, ann->d_output, (num_total_neurons + num_neuron_append) * sizeof(float), cudaMemcpyDeviceToHost));
    ann->output_cuda = host_output;
    
    if(run_type == 1){
        CUDA_CHECK(cudaFree(ann->d_output));
        CUDA_CHECK(cudaFree(ann->d_weights));
        ann->d_output = NULL;
        ann->d_weights = NULL;
    }

    return host_output + (num_total_neurons + num_neuron_append - num_output);
}

void genann_train_cuda(genann *ann, float const *inputs, float const *desired_outputs, float learning_rate){
    // Get the outputs of each activation
    genann_run_cuda(ann, inputs, INTERNAL);

    int h, j, k;

    // Set the output layer deltas
    {
        float const *o = ann->d_output + (ann->inputs + 1) + (ann->hidden + 1) * ann->hidden_layers; /* First output. */
        float *d = ann->delta + ann->hidden * ann->hidden_layers;                      /* First delta. */
        float const *t = desired_outputs;                                              /* First desired output. */

        /* Set output layer deltas. */
    }
    
}