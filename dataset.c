#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "dataset.h"

POINT points[NUMSAMPLES];

void shuffle()
{
    for (int i = 0; i < NUMSAMPLES; i++)
    {
        int index = i * ((double)rand() / RAND_MAX);
        POINT point = points[i];
        points[i] = points[index];
        points[index] = point;
    }
}

// 创建NUMSAMPLES个参数，按照原型来创建
void classifyCircleData()
{
    double radius = 5;
    // 创建内部圆上的点
    for (int i = 0; i < NUMSAMPLES / 2; i++)
    {
        double r = 0.5 * radius * rand() / RAND_MAX;   // 生成随机的半径
        double angle = 2.0 * M_PI * rand() / RAND_MAX; // 生成随机的角度
        points[i].x = r * cos(angle);
        points[i].y = r * sin(angle);
        points[i].label = 1;
    }
    // 创建外部圆上的点
    for (int i = NUMSAMPLES / 2; i < NUMSAMPLES; i++)
    {
        double r = 0.7 * radius + 0.3 * radius * rand() / RAND_MAX; // 生成随机的半径
        double angle = 2.0 * M_PI * rand() / RAND_MAX;              // 生成随机的角度
        points[i].x = r * cos(angle);
        points[i].y = r * sin(angle);
        points[i].label = -1;
    }
    shuffle();
}
