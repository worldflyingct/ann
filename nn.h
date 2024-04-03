#ifndef __NN_H__
#define __NN_H__

#include "config.h"

typedef struct
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
    double output;
    double inputDer;
    double outputDer;
    double accInputDer;
    int numAccumulatedDers;
    double totalInput;
} NODE;

double square(double output, double target);
double squareder(double output, double target);
double tanhder(double x); // tanh的倒数
double activation(double x);
double activationder(double x);
double outlayeractivation(double x);
double outlayeractivationder(double x);
void buildNetwork();
void forwardProp(POINT point);
void backProp(POINT point);
void updateWeights();

#endif
