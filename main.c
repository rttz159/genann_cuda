#include <stdio.h>
#include "genann.h"

int main()
{
    genann *ann = genann_init(2, 1, 2, 2);
    double input[] = {1, 2};
    const double *output = genann_run(ann, input);
    printf("Results: \n");
    for (int i = 0; i < 2; i++)
    {
        printf("%f\n", *output);
        output++;
    }
}