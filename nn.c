#include <stdlib.h>
#include "config.h"
#include "nn.h"

NODE network[HIDDEN_LAYER_NUM][LAYER_NEURON_NUM];
NODE outputlayer;

double square(double output, double target)
{
    return (output - target) * (output - target) / 2;
}

double squareder(double output, double target)
{
    return output - target;
}

double activation(double x)
{
#if ACTIVATIONFUNCTION == RELU
    if (x > 0)
    {
        return x;
    }
    else
    {
        return 0;
    }
#endif
}

double activationder(double x)
{
#if ACTIVATIONFUNCTION == RELU
    if (x > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
#endif
}

void buildNetwork()
{
    for (int i = 0; i < HIDDEN_LAYER_NUM; i++)
    {
        for (int j = 0; j < LAYER_NEURON_NUM; j++)
        {
            network[i][j].bias = 0.1;
            for (int k = 0; k < LAYER_NEURON_NUM; k++)
            {
                network[i][j].link[k].weight = 1.0 * rand() / RAND_MAX - 0.5;
                network[i][j].link[k].errorDer = 0;
                network[i][j].link[k].accErrorDer = 0;
                network[i][j].link[k].numAccumulatedDers = 0;
            }
            network[i][j].inputDer = 0;
            network[i][j].outputDer = 0;
            network[i][j].accInputDer = 0;
            network[i][j].numAccumulatedDers = 0;
        }
    }
    outputlayer.bias = 0.1;
    for (int k = 0; k < LAYER_NEURON_NUM; k++)
    {
        outputlayer.link[k].weight = 1.0 * rand() / RAND_MAX - 0.5;
        outputlayer.link[k].errorDer = 0;
        outputlayer.link[k].accErrorDer = 0;
        outputlayer.link[k].numAccumulatedDers = 0;
    }
    outputlayer.inputDer = 0;
    outputlayer.outputDer = 0;
    outputlayer.accInputDer = 0;
    outputlayer.numAccumulatedDers = 0;
}
