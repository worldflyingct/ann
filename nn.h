#ifndef __NN_H__
#define __NN_H__

#include "config.h"

typedef struct nn
{
    double weight;
    double errorDer;
    double accErrorDer;
    int numAccumulatedDers;
} LINK;

typedef struct
{
    double bias;
    LINK link[LAYER_NEURON_NUM];
    double out;
    double inputDer;
    double outputDer;
    double accInputDer;
    int numAccumulatedDers;
    double totalInput;
} NODE;

double square(double output, double target);
double squareder(double output, double target);
double activation(double x);
double activationder(double x);
void buildNetwork();

#endif
