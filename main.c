#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "config.h"
#include "dataset.h"
#include "nn.h"

extern NODE network[HIDDEN_LAYER_NUM][LAYER_NEURON_NUM];
extern NODE outputlayer;
extern POINT points[NUMSAMPLES];

void forwardProp(POINT point)
{
    for (int i = 0; i < LAYER_NEURON_NUM; i++)
    { // 隐藏层第一层
        network[0][i].totalInput = network[0][i].bias;
        network[0][i].totalInput += network[0][i].link[0].weight * point.x;
        network[0][i].totalInput += network[0][i].link[1].weight * point.y;
        network[0][i].output = activation(network[0][i].totalInput);
    }
    for (int i = 1; i < HIDDEN_LAYER_NUM; i++)
    { // 隐藏层非第一层
        for (int j = 0; j < LAYER_NEURON_NUM; j++)
        {
            network[i][j].totalInput = network[i][j].bias;
            for (int k = 0; k < LAYER_NEURON_NUM; k++)
            {
                network[i][j].totalInput += network[i][j].link[k].weight * network[i - 1][k].output;
            }
            network[i][j].output = activation(network[i][j].totalInput);
        }
    }
    // 输出层
    outputlayer.totalInput = outputlayer.bias;
    for (int i = 0; i < LAYER_NEURON_NUM; i++)
    {
        outputlayer.totalInput += outputlayer.link[i].weight * network[HIDDEN_LAYER_NUM - 1][i].output;
    }
    outputlayer.output = outlayeractivation(outputlayer.totalInput);
}

void backProp(POINT point)
{
    // 清空所有节点的outputDer
    for (int i = 0; i < HIDDEN_LAYER_NUM; i++)
    {
        for (int j = 0; j < LAYER_NEURON_NUM; j++)
        {
            network[i][j].outputDer = 0;
        }
    }
    // 输出层
    outputlayer.outputDer = squareder(outputlayer.output, point.label); // 目标和结果的差距
    outputlayer.inputDer = outputlayer.outputDer * outlayeractivationder(outputlayer.totalInput);
    outputlayer.accInputDer += outputlayer.inputDer;
    outputlayer.numAccumulatedDers++;
    for (int i = 0; i < LAYER_NEURON_NUM; i++)
    {
        outputlayer.link[i].errorDer = outputlayer.inputDer * network[HIDDEN_LAYER_NUM - 1][i].output;
        outputlayer.link[i].accErrorDer += outputlayer.link[i].errorDer;
        outputlayer.link[i].numAccumulatedDers++;
        network[HIDDEN_LAYER_NUM - 1][i].outputDer += outputlayer.link[i].weight * outputlayer.inputDer;
    }
    for (int i = HIDDEN_LAYER_NUM - 1; i >= 1; i--)
    { // 隐藏层非第一层
        for (int j = 0; j < LAYER_NEURON_NUM; j++)
        {
            network[i][j].inputDer = network[i][j].outputDer * activationder(network[i][j].totalInput);
            network[i][j].accInputDer += network[i][j].inputDer;
            network[i][j].numAccumulatedDers++;
            for (int k = 0; k < LAYER_NEURON_NUM; k++)
            {
                network[i][j].link[k].errorDer = network[i][j].inputDer * network[i - 1][k].output;
                network[i][j].link[k].accErrorDer += network[i][j].link[k].errorDer;
                network[i][j].link[k].numAccumulatedDers++;
                network[i - 1][k].outputDer += network[i][j].link[k].weight * network[i][j].inputDer;
            }
        }
    }
    // 隐藏层第一层
    for (int i = 0; i < LAYER_NEURON_NUM; i++)
    {
        network[0][i].inputDer = network[0][i].outputDer * activationder(network[0][i].totalInput);
        network[0][i].accInputDer += network[0][i].inputDer;
        network[0][i].numAccumulatedDers++;
        network[0][i].link[0].errorDer = network[0][i].inputDer * point.x;
        network[0][i].link[0].accErrorDer += network[0][i].link[0].errorDer;
        network[0][i].link[0].numAccumulatedDers++;
        network[0][i].link[1].errorDer = network[0][i].inputDer * point.y;
        network[0][i].link[1].accErrorDer += network[0][i].link[1].errorDer;
        network[0][i].link[1].numAccumulatedDers++;
    }
}

void updateWeights()
{
    for (int i = 0; i < LAYER_NEURON_NUM; i++)
    { // 隐藏层第一层
        if (network[0][i].numAccumulatedDers > 0)
        {
            network[0][i].bias -= LEARNINGRATE * network[0][i].accInputDer / network[0][i].numAccumulatedDers;
            network[0][i].accInputDer = 0;
            network[0][i].numAccumulatedDers = 0;
        }
        if (network[0][i].link[0].numAccumulatedDers > 0)
        {
            network[0][i].link[0].weight -= LEARNINGRATE * network[0][i].link[0].accErrorDer / network[0][i].link[0].numAccumulatedDers;
            network[0][i].link[0].accErrorDer = 0;
            network[0][i].link[0].numAccumulatedDers = 0;
        }
        if (network[0][i].link[1].numAccumulatedDers > 0)
        {
            network[0][i].link[1].weight -= LEARNINGRATE * network[0][i].link[1].accErrorDer / network[0][i].link[1].numAccumulatedDers;
            network[0][i].link[1].accErrorDer = 0;
            network[0][i].link[1].numAccumulatedDers = 0;
        }
    }
    for (int i = 1; i < HIDDEN_LAYER_NUM; i++)
    { // 隐藏层非第一层
        for (int j = 0; j < LAYER_NEURON_NUM; j++)
        {
            if (network[i][j].numAccumulatedDers > 0)
            {
                network[i][j].bias -= LEARNINGRATE * network[i][j].accInputDer / network[i][j].numAccumulatedDers;
                network[i][j].accInputDer = 0;
                network[i][j].numAccumulatedDers = 0;
            }
            for (int k = 0; k < LAYER_NEURON_NUM; k++)
            {
                if (network[i][j].link[k].numAccumulatedDers > 0)
                {
                    network[i][j].link[k].weight -= LEARNINGRATE * network[i][j].link[k].accErrorDer / network[i][j].link[k].numAccumulatedDers;
                    network[i][j].link[k].accErrorDer = 0;
                    network[i][j].link[k].numAccumulatedDers = 0;
                }
            }
        }
    }
    // 输出层
    if (outputlayer.numAccumulatedDers > 0)
    {
        outputlayer.bias -= LEARNINGRATE * outputlayer.accInputDer / outputlayer.numAccumulatedDers;
        outputlayer.accInputDer = 0;
        outputlayer.numAccumulatedDers = 0;
    }
    for (int i = 0; i < LAYER_NEURON_NUM; i++)
    {
        if (outputlayer.link[i].numAccumulatedDers > 0)
        {
            outputlayer.link[i].weight -= LEARNINGRATE * outputlayer.link[i].accErrorDer / outputlayer.link[i].numAccumulatedDers;
            outputlayer.link[i].accErrorDer = 0;
            outputlayer.link[i].numAccumulatedDers = 0;
        }
    }
}

double getLoss(int mode) // 0代表训练集，1代表测试集
{
    double loss = 0;
    if (mode)
    {
        for (int i = NUMSAMPLES / 2; i < NUMSAMPLES; i++)
        {
            forwardProp(points[i]);
            loss += square(outputlayer.output, points[i].label);
        }
    }
    else
    {
        for (int i = 0; i < NUMSAMPLES / 2; i++)
        {
            forwardProp(points[i]);
            loss += square(outputlayer.output, points[i].label);
        }
    }
    return loss / (NUMSAMPLES / 2);
}

void training()
{
    for (int i = 0; i < NUMSAMPLES / 2; i++)
    {
        forwardProp(points[i]);
        backProp(points[i]);
        if ((i + 1) % BATCHSIZE == 0)
        {
            updateWeights();
        }
    }
    double lossTrain = getLoss(0);
    double lossTest = getLoss(1);
    printf("lossTrain:%f,lossTest:%f\n", lossTrain, lossTest);
}

int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));
    classifyCircleData();
    buildNetwork();
    double lossTrain = getLoss(0);
    double lossTest = getLoss(1);
    printf("lossTrain:%f,lossTest:%f\n", lossTrain, lossTest);
    for (int i = 0; i < 100; i++)
    {
        training();
    }
    return 0;
}
