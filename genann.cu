/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#include "genann.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifndef genann_act
#define genann_act_hidden genann_act_hidden_indirect
#define genann_act_output genann_act_output_indirect
#else
#define genann_act_hidden genann_act
#define genann_act_output genann_act
#endif

#define LOOKUP_SIZE 4096

float genann_act_hidden_indirect(const struct genann *ann, float a)
{
    return ann->activation_hidden(ann, a);
}

float genann_act_output_indirect(const struct genann *ann, float a)
{
    return ann->activation_output(ann, a);
}

const float sigmoid_dom_min = -15.0;
const float sigmoid_dom_max = 15.0;
float interval;
float lookup[LOOKUP_SIZE];

#ifdef __GNUC__
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define unused __attribute__((unused))
#else
#define likely(x) x
#define unlikely(x) x
#define unused
#pragma warning(disable : 4996) /* For fscanf */
#endif

float genann_act_sigmoid(const genann *ann unused, float a)
{
    if (a < -45.0)
        return 0;
    if (a > 45.0)
        return 1;
    return 1.0 / (1 + exp(-a));
}

void genann_init_sigmoid_lookup(const genann *ann)
{
    const float f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;
    int i;

    interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
    for (i = 0; i < LOOKUP_SIZE; ++i)
    {
        lookup[i] = genann_act_sigmoid(ann, sigmoid_dom_min + f * i);
    }
}

float genann_act_sigmoid_cached(const genann *ann unused, float a)
{
    assert(!isnan(a));

    if (a < sigmoid_dom_min)
        return lookup[0];
    if (a >= sigmoid_dom_max)
        return lookup[LOOKUP_SIZE - 1];

    size_t j = (size_t)((a - sigmoid_dom_min) * interval + 0.5);

    /* Because floating point... */
    if (unlikely(j >= LOOKUP_SIZE))
        return lookup[LOOKUP_SIZE - 1];

    return lookup[j];
}

float genann_act_linear(const struct genann *ann unused, float a)
{
    return a;
}

float genann_act_threshold(const struct genann *ann unused, float a)
{
    return a > 0;
}

genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs)
{
    if (hidden_layers < 0)
        return 0;
    if (inputs < 1)
        return 0;
    if (outputs < 1)
        return 0;
    if (hidden_layers > 0 && hidden < 1)
        return 0;

    const int hidden_weights = hidden_layers ? (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden : 0;
    const int output_weights = (hidden_layers ? (hidden + 1) : (inputs + 1)) * outputs;
    const int total_weights = (hidden_weights + output_weights);

    const int total_neurons = (inputs + hidden * hidden_layers + outputs);

    /* Allocate extra size for weights, outputs, and deltas. */
    const int size = sizeof(genann) + sizeof(float) * (total_weights + total_neurons + (total_neurons - inputs));
    genann *ret = (genann *) malloc(size);
    if (!ret)
        return 0;

    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;

    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;

    /* Set pointers. */
    ret->weight = (float *)((char *)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    genann_randomize(ret);

    ret->activation_hidden = genann_act_sigmoid_cached;
    ret->activation_output = genann_act_sigmoid_cached;

    ret->activation_hidden_type = GENANN_ACT_SIGMOID;
    ret->activation_output_type = GENANN_ACT_SIGMOID;

    genann_init_sigmoid_lookup(ret);

    return ret;
}

genann *genann_read(FILE *in)
{
    int inputs, hidden_layers, hidden, outputs;
    int rc;

    errno = 0;
    rc = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);
    if (rc < 4 || errno != 0)
    {
        perror("fscanf");
        return NULL;
    }

    genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i)
    {
        errno = 0;
        rc = fscanf(in, " %f", ann->weight + i);
        if (rc < 1 || errno != 0)
        {
            perror("fscanf");
            genann_free(ann);

            return NULL;
        }
    }

    return ann;
}

genann *genann_copy(genann const *ann)
{
    const int size = sizeof(genann) + sizeof(float) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
    genann *ret = (genann *) malloc(size);
    if (!ret)
        return 0;

    memcpy(ret, ann, size);

    /* Set pointers. */
    ret->weight = (float *)((char *)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    return ret;
}

void genann_randomize(genann *ann)
{
    int i;
    for (i = 0; i < ann->total_weights; ++i)
    {
        float r = GENANN_RANDOM();
        /* Sets weights from -0.5 to 0.5. */
        ann->weight[i] = r - 0.5;
    }
}

void genann_free(genann *ann)
{
    if (ann->d_output)
    {
        cudaFree(ann->d_output);
        ann->d_output = NULL;
    }

    if (ann->d_weights)
    {
        cudaFree(ann->d_weights);
        ann->d_weights = NULL;
    }

    if (ann->output_cuda)
        free(ann->output_cuda);
    /* The weight, output, and delta pointers go to the same buffer. */
    free(ann);
}

float const *genann_run(genann const *ann, float const *inputs)
{
    float const *w = ann->weight;
    float *o = ann->output + ann->inputs;
    float const *i = ann->output;

    /* Copy the inputs to the scratch area, where we also store each neuron's
     * output, for consistency. This way the first layer isn't a special case. */
    memcpy(ann->output, inputs, sizeof(float) * ann->inputs);

    int h, j, k;

    if (!ann->hidden_layers)
    {
        float *ret = o;
        for (j = 0; j < ann->outputs; ++j)
        {
            float sum = *w++ * -1.0;
            for (k = 0; k < ann->inputs; ++k)
            {
                sum += *w++ * i[k];
            }
            *o++ = genann_act_output(ann, sum);
        }

        return ret;
    }

    /* Figure input layer */
    for (j = 0; j < ann->hidden; ++j)
    {
        float sum = *w++ * -1.0;
        for (k = 0; k < ann->inputs; ++k)
        {
            sum += *w++ * i[k];
        }
        *o++ = genann_act_hidden(ann, sum);
    }

    i += ann->inputs;

    /* Figure hidden layers, if any. */
    for (h = 1; h < ann->hidden_layers; ++h)
    {
        for (j = 0; j < ann->hidden; ++j)
        {
            float sum = *w++ * -1.0;
            for (k = 0; k < ann->hidden; ++k)
            {
                sum += *w++ * i[k];
            }
            *o++ = genann_act_hidden(ann, sum);
        }

        i += ann->hidden;
    }

    float const *ret = o;

    /* Figure output layer. */
    for (j = 0; j < ann->outputs; ++j)
    {
        float sum = *w++ * -1.0;
        for (k = 0; k < ann->hidden; ++k)
        {
            sum += *w++ * i[k];
        }
        *o++ = genann_act_output(ann, sum);
    }

    /* Sanity check that we used all weights and wrote all outputs. */
    assert(w - ann->weight == ann->total_weights);
    assert(o - ann->output == ann->total_neurons);

    return ret;
}

void genann_train(genann const *ann, float const *inputs, float const *desired_outputs, float learning_rate)
{
    /* To begin with, we must run the network forward. */
    genann_run(ann, inputs);

    int h, j, k;

    /* First set the output layer deltas. */
    {
        float const *o = ann->output + ann->inputs + ann->hidden * ann->hidden_layers; /* First output. */
        float *d = ann->delta + ann->hidden * ann->hidden_layers;                      /* First delta. */
        float const *t = desired_outputs;                                              /* First desired output. */

        /* Set output layer deltas. */
        if (genann_act_output == genann_act_linear ||
            ann->activation_output == genann_act_linear)
        {
            for (j = 0; j < ann->outputs; ++j)
            {
                *d++ = *t++ - *o++;
            }
        }
        else
        {
            for (j = 0; j < ann->outputs; ++j)
            {
                *d++ = (*t - *o) * *o * (1.0 - *o);
                ++o;
                ++t;
            }
        }
    }

    /* Set hidden layer deltas, start on last layer and work backwards. */
    /* Note that loop is skipped in the case of hidden_layers == 0. */
    for (h = ann->hidden_layers - 1; h >= 0; --h)
    {

        /* Find first output and delta in this layer. */
        float const *o = ann->output + ann->inputs + (h * ann->hidden);
        float *d = ann->delta + (h * ann->hidden);

        /* Find first delta in following layer (which may be hidden or output). */
        float const *const dd = ann->delta + ((h + 1) * ann->hidden);

        /* Find first weight in following layer (which may be hidden or output). */
        float const *const ww = ann->weight + ((ann->inputs + 1) * ann->hidden) + ((ann->hidden + 1) * ann->hidden * (h));

        for (j = 0; j < ann->hidden; ++j)
        {

            float delta = 0;

            for (k = 0; k < (h == ann->hidden_layers - 1 ? ann->outputs : ann->hidden); ++k)
            {
                const float forward_delta = dd[k];
                const int windex = k * (ann->hidden + 1) + (j + 1);
                const float forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
            }

            *d = *o * (1.0 - *o) * delta;
            ++d;
            ++o;
        }
    }

    /* Train the outputs. */
    {
        /* Find first output delta. */
        float const *d = ann->delta + ann->hidden * ann->hidden_layers; /* First output delta. */

        /* Find first weight to first output delta. */
        float *w = ann->weight + (ann->hidden_layers
                                      ? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * ann->hidden * (ann->hidden_layers - 1))
                                      : (0));

        /* Find first output in previous layer. */
        float const *const i = ann->output + (ann->hidden_layers
                                                  ? (ann->inputs + (ann->hidden) * (ann->hidden_layers - 1))
                                                  : 0);

        /* Set output layer weights. */
        for (j = 0; j < ann->outputs; ++j)
        {
            *w++ += *d * learning_rate * -1.0;
            for (k = 1; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; ++k)
            {
                *w++ += *d * learning_rate * i[k - 1];
            }

            ++d;
        }

        assert(w - ann->weight == ann->total_weights);
    }

    /* Train the hidden layers. */
    for (h = ann->hidden_layers - 1; h >= 0; --h)
    {

        /* Find first delta in this layer. */
        float const *d = ann->delta + (h * ann->hidden);

        /* Find first input to this layer. */
        float const *i = ann->output + (h
                                            ? (ann->inputs + ann->hidden * (h - 1))
                                            : 0);

        /* Find first weight to this layer. */
        float *w = ann->weight + (h
                                      ? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * (ann->hidden) * (h - 1))
                                      : 0);

        for (j = 0; j < ann->hidden; ++j)
        {
            *w++ += *d * learning_rate * -1.0;
            for (k = 1; k < (h == 0 ? ann->inputs : ann->hidden) + 1; ++k)
            {
                *w++ += *d * learning_rate * i[k - 1];
            }
            ++d;
        }
    }
}

void genann_write(genann const *ann, FILE *out)
{
    fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i)
    {
        fprintf(out, " %.20e", ann->weight[i]);
    }
}
