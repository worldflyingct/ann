#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "nn.h"

NODE network[HIDDEN_LAYER_NUM][LAYER_NEURON_NUM];
NODE outputlayer;

double square(double output, double target)
{
    double r = output - target;
    return r * r / 2;
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

double outlayeractivation(double x)
{
#if OUTLAYERACTIVATIONFUNCTION == TANH
    return tanh(x);
#endif
}

double outlayeractivationder(double x)
{
#if OUTLAYERACTIVATIONFUNCTION == TANH
    // tanh的倒数
    double y = tanh(x);
    return 1 - y * y;
#endif
}

void buildNetwork()
{
    for (int i = 0; i < LAYER_NEURON_NUM; i++)
    {
        network[0][i].bias = 0.1;
        network[0][i].link[0].weight = 1.0 * rand() / RAND_MAX - 0.5;
        network[0][i].link[0].errorDer = 0;
        network[0][i].link[0].accErrorDer = 0;
        network[0][i].link[0].numAccumulatedDers = 0;
        network[0][i].link[1].weight = 1.0 * rand() / RAND_MAX - 0.5;
        network[0][i].link[1].errorDer = 0;
        network[0][i].link[1].accErrorDer = 0;
        network[0][i].link[1].numAccumulatedDers = 0;
        network[0][i].inputDer = 0;
        network[0][i].outputDer = 0;
        network[0][i].accInputDer = 0;
        network[0][i].numAccumulatedDers = 0;
    }
    for (int i = 1; i < HIDDEN_LAYER_NUM; i++)
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
