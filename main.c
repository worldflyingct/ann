#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "config.h"
#include "dataset.h"
#include "nn.h"

extern POINT points[NUMSAMPLES];

double getLoss(int mode) // 0代表训练集，1代表测试集
{
    double loss = 0;
    if (mode)
    {
        for (int i = NUMSAMPLES / 2; i < NUMSAMPLES; i++)
        {
            forwardProp(points[i]);
            loss += square(getOutPut(), points[i].label);
        }
    }
    else
    {
        for (int i = 0; i < NUMSAMPLES / 2; i++)
        {
            forwardProp(points[i]);
            loss += square(getOutPut(), points[i].label);
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
