#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "dataset.h"
#include "nn.h"

int networkShape[] = {2, 8, 8, 8, 8, 8, 8, 1};
NODE **network;

double getOutPut()
{
    return network[sizeof(networkShape) / sizeof(int) - 1][0].output;
}

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
#elif ACTIVATIONFUNCTION == TANH
    return tanh(x);
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
#elif ACTIVATIONFUNCTION == TANH
    // tanh的倒数
    double y = tanh(x);
    return 1 - y * y;
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
    network = (PPNODE)malloc((sizeof(networkShape) / sizeof(int)) * sizeof(PNODE));
    // 输入层
    network[0] = (PNODE)malloc(networkShape[0] * sizeof(NODE));
    // 隐藏层与输出层
    for (int i = 1, leni = sizeof(networkShape) / sizeof(int); i < leni; i++)
    {
        network[i] = (PNODE)malloc(networkShape[i] * sizeof(NODE));
        int prenodeNum = networkShape[i - 1];
        for (int j = 0, lenj = networkShape[i]; j < lenj; j++)
        {
            network[i][j].link = (PLINK)malloc(prenodeNum * sizeof(LINK));
        }
    }
    // 输入层
    for (int i = 0; i < networkShape[0]; i++)
    {
        network[0][i].bias = 0.1;
    }
    // 隐藏层与输出层
    for (int i = 1, leni = sizeof(networkShape) / sizeof(int); i < leni; i++)
    {
        for (int j = 0, lenj = networkShape[i]; j < lenj; j++)
        {
            network[i][j].bias = 0.1;
            network[i][j].inputDer = 0;
            network[i][j].outputDer = 0;
            network[i][j].accInputDer = 0;
            network[i][j].numAccumulatedDers = 0;
            for (int k = 0, lenk = networkShape[i - 1]; k < lenk; k++)
            {
                network[i][j].link[k].weight = (double)rand() / RAND_MAX - 0.5;
                network[i][j].link[k].errorDer = 0;
                network[i][j].link[k].accErrorDer = 0;
                network[i][j].link[k].numAccumulatedDers = 0;
            }
        }
    }
}

void forwardProp(POINT point)
{
    int outlayerNum = sizeof(networkShape) / sizeof(int) - 1; // 输出层所在层
    // 输入层
    network[0][0].output = point.x;
    network[0][1].output = point.y;
    // 隐藏层
    for (int i = 1, leni = outlayerNum; i < leni; i++)
    {
        for (int j = 0, lenj = networkShape[i]; j < lenj; j++)
        {
            network[i][j].totalInput = network[i][j].bias;
            for (int k = 0, lenk = networkShape[i - 1]; k < lenk; k++)
            {
                network[i][j].totalInput += network[i][j].link[k].weight * network[i - 1][k].output;
            }
            network[i][j].output = activation(network[i][j].totalInput);
        }
    }
    // 输出层
    for (int i = 0, leni = networkShape[outlayerNum]; i < leni; i++)
    {
        network[outlayerNum][i].totalInput = network[outlayerNum][i].bias;
        for (int j = 0, lenj = networkShape[outlayerNum - 1]; j < lenj; j++)
        {
            network[outlayerNum][i].totalInput += network[outlayerNum][i].link[j].weight * network[outlayerNum - 1][j].output;
        }
        network[outlayerNum][i].output = outlayeractivation(network[outlayerNum][i].totalInput);
    }
}

void backProp(POINT point)
{
    // 清空所有节点的outputDer
    for (int i = 0, leni = sizeof(networkShape) / sizeof(int); i < leni; i++)
    {
        for (int j = 0; j < networkShape[i]; j++)
        {
            network[i][j].outputDer = 0;
        }
    }
    int outlayerNum = sizeof(networkShape) / sizeof(int) - 1; // 输出层所在层
    // 输出层
    for (int i = 0, leni = networkShape[outlayerNum]; i < leni; i++)
    {
        network[outlayerNum][i].outputDer = squareder(network[outlayerNum][i].output, point.label); // 目标和结果的差距
        network[outlayerNum][i].inputDer = network[outlayerNum][i].outputDer * outlayeractivationder(network[outlayerNum][i].totalInput);
        network[outlayerNum][i].accInputDer += network[outlayerNum][i].inputDer;
        network[outlayerNum][i].numAccumulatedDers++;
        for (int j = 0, lenj = networkShape[outlayerNum]; j < lenj; j++)
        {
            network[outlayerNum][i].link[i].errorDer = network[outlayerNum][i].inputDer * network[outlayerNum - 1][i].output;
            network[outlayerNum][i].link[i].accErrorDer += network[outlayerNum][i].link[i].errorDer;
            network[outlayerNum][i].link[i].numAccumulatedDers++;
            network[outlayerNum - 1][i].outputDer += network[outlayerNum][i].link[i].weight * network[outlayerNum][i].inputDer;
        }
    }
    // 隐藏层
    for (int i = outlayerNum; i > 0; i--)
    {
        for (int j = 0; j < networkShape[i]; j++)
        {
            network[i][j].inputDer = network[i][j].outputDer * activationder(network[i][j].totalInput);
            network[i][j].accInputDer += network[i][j].inputDer;
            network[i][j].numAccumulatedDers++;
            for (int k = 0; k < networkShape[i - 1]; k++)
            {
                network[i][j].link[k].errorDer = network[i][j].inputDer * network[i - 1][k].output;
                network[i][j].link[k].accErrorDer += network[i][j].link[k].errorDer;
                network[i][j].link[k].numAccumulatedDers++;
                network[i - 1][k].outputDer += network[i][j].link[k].weight * network[i][j].inputDer;
            }
        }
    }
}

void updateWeights()
{
    // 隐藏层与输出层
    for (int i = 1; i < sizeof(networkShape) / sizeof(int); i++)
    {
        for (int j = 0; j < networkShape[i]; j++)
        {
            if (network[i][j].numAccumulatedDers > 0)
            {
                network[i][j].bias -= LEARNINGRATE * network[i][j].accInputDer / network[i][j].numAccumulatedDers;
                network[i][j].accInputDer = 0;
                network[i][j].numAccumulatedDers = 0;
            }
            for (int k = 0; k < networkShape[i - 1]; k++)
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
}
