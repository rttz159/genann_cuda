#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include "genann_cuda.cuh"
#include "genann.h"

int main()
{
    printf("GENANN CPU/CUDA Comparison Test\n");
    printf("--------------------------------\n");

    srand(time(0));

    // --- Neural Network Configuration ---
    const int inputs = 2;
    const int hidden_layers = 1;
    const int hidden_neurons = 4;
    const int outputs = 1;

    // --- XOR Training Data ---
    // Input patterns for XOR
    float input_data[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}};

    // Desired outputs for XOR
    float output_data[4][1] = {
        {0},
        {1},
        {1},
        {0}};

    // --- Initialize genann (CPU version) ---
    printf("Initializing CPU neural network...\n");
    genann *ann_cpu = genann_init(inputs, hidden_layers, hidden_neurons, outputs);
    if (!ann_cpu)
    {
        fprintf(stderr, "Error: Could not initialize CPU neural network.\n");
        return 1;
    }
    printf("CPU network initialized. Inputs: %d, Hidden Layers: %d, Hidden Neurons: %d, Outputs: %d\n",
           ann_cpu->inputs, ann_cpu->hidden_layers, ann_cpu->hidden, ann_cpu->outputs);

    // --- Test the CPU network ---
    printf("\nTesting CPU network results:\n");
    for (int i = 0; i < 4; ++i)
    {
        const float *output_cpu = genann_run(ann_cpu, input_data[i]);
        printf("Input: [%.1f, %.1f], Desired: %.1f, CPU Output: %.4f\n",
               input_data[i][0], input_data[i][1], output_data[i][0], output_cpu[0]);
    }

    // --- Test the CUDA network ---
    printf("\nTesting CUDA network results (should be similar to CPU output):\n");
    for (int i = 0; i < 4; ++i)
    {
        // We need a non-const pointer for genann_run_cuda as it expects float*
        float *input_cuda = (float *)malloc(inputs * sizeof(float));
        if (!input_cuda)
        {
            fprintf(stderr, "Error: Malloc failed for input_cuda.\n");
            genann_free(ann_cpu);
            return 1;
        }
        memcpy(input_cuda, input_data[i], inputs * sizeof(float));

        float *output_cuda = genann_run_cuda(ann_cpu, input_cuda); // Call CUDA function

        printf("Input: [%.1f, %.1f], Desired: %.1f, CUDA Output: %.4f\n",
               input_data[i][0], input_data[i][1], output_data[i][0], output_cuda[0]);

        // Compare CPU and CUDA outputs
        // Due to floating point precision and different computation paths (CPU vs GPU),
        // outputs might not be *exactly* identical, but should be very close.
        float diff = fabs(genann_run(ann_cpu, input_data[i])[0] - output_cuda[0]);
        printf("   Difference (CPU vs CUDA): %.8f\n", diff);
        if (diff > 1e-4)
        { // A small tolerance for floating point differences
            printf("   WARNING: Significant difference detected between CPU and CUDA output for this input.\n");
        }

        free(input_cuda);
        free(output_cuda); // Free memory allocated by genann_run_cuda
    }

    // --- Clean up ---
    printf("\nCleaning up...\n");
    genann_free(ann_cpu);
    printf("Cleanup complete. Test finished.\n");

    return 0;
}
