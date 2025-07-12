#ifndef __GENANN_CUDA_CUH__
#define __GENANN_CUDA_CUH__

#include <cuda_runtime.h>
#include "genann.h"

#define TILE_WIDTH 32
#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

__device__ float genann_act_linear_cuda(float a){
    return a;
}

__device__ float genann_act_threshold_cuda(float a){
    return a > 0;
}

__device__ float genann_act_sigmoid_cuda(float a)
{
    if (a < -45.0)
        return 0;
    if (a > 45.0)
        return 1;
    return 1.0 / (1 + exp(-a));
}

#ifdef __cplusplus
extern "C" {
#endif

float *genann_run_cuda(genann *ann, float const *inputs, enum GenannRunType run_type);
void genann_train_cuda(genann *ann, float const *inputs, float const *desired_outputs, float learning_rate);

#ifdef __cplusplus
}
#endif

#endif