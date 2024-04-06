#ifndef __NN_H__
#define __NN_H__

#include "config.h"

typedef struct LINK
{
    double weight;
    double errorDer;
    double accErrorDer;
    int numAccumulatedDers;
} LINK;
typedef LINK *PLINK;

typedef struct NODE
{
    double bias;
    PLINK link;
    double output;
    double inputDer;
    double outputDer;
    double accInputDer;
    int numAccumulatedDers;
    double totalInput;
} NODE;
typedef NODE *PNODE;
typedef PNODE *PPNODE;

double getOutPut();
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
