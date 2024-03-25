#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LAYER_NEURON_NUM 6
#define HIDDEN_LAYER_NUM 8

struct NEURON
{
    double value;
    double bias;
    double weight[HIDDEN_LAYER_NUM];
};
struct NEURON neurons[HIDDEN_LAYER_NUM][LAYER_NEURON_NUM];
int out_weight[LAYER_NEURON_NUM]; // 最后一级的权重

double calc(double x, double y)
{
    for (int i = 0; i < LAYER_NEURON_NUM; i++)
    {
        double sum = 0;
        sum += x * neurons[0][i].weight[0];
        sum += y * neurons[0][i].weight[1];
        sum += neurons[0][i].bias;
        neurons[0][i].value = sum > 0 ? sum : 0;
    }
    for (int i = 1; i < HIDDEN_LAYER_NUM; i++)
    {
        for (int j = 0; j < LAYER_NEURON_NUM; j++)
        {
            double sum = 0;
            for (int k = 0; k < LAYER_NEURON_NUM; k++)
            {
                sum += neurons[i - 1][k].value * neurons[i][j].weight[k];
            }
            sum += neurons[i][j].bias;
            neurons[i][i].value = sum > 0 ? sum : 0;
        }
    }
    double out = 0;
    for (int i = 0; i < LAYER_NEURON_NUM; i++)
    {
        out += neurons[HIDDEN_LAYER_NUM - 1][i].value * out_weight[i];
    }
    return out;
}

void init_neurons()
{
    for (int i = 0; i < HIDDEN_LAYER_NUM; i++)
    {
        for (int j = 0; j < LAYER_NEURON_NUM; j++)
        {
            neurons[i][j].bias = 0.1;
            neurons[i][j].value = 0;
            for (int k = 0; k < LAYER_NEURON_NUM; k++)
            {
                neurons[i][j].weight[k] = 2.0 * rand() / RAND_MAX - 1;
            }
            out_weight[j] = 2.0 * rand() / RAND_MAX - 1;
        }
    }
}

double trains_data[13][13];
void init_trains_data()
{
    memset(trains_data, 0, sizeof(trains_data));
    trains_data[11][6] = 1;
    trains_data[10][3] = -1;
}

double check_data()
{
    int right = 0;
    int total = 0;
    for (int x = 0; x < 13; x++)
    {
        for (int y = 0; y < 13; y++)
        {
            if (trains_data[x][y] != 0)
            {
                total++;
                double r = calc(x - 6, y - 6);

                if ((r > 0 && trains_data[x][y] > 0) || (r < 0 && trains_data[x][y] < 0))
                {
                    right++;
                }
            }
        }
    }
    return 100.0 * right / total;
}

int training()
{
    double r = check_data();
    while (r < 100.0)
    {
        int i = rand() % HIDDEN_LAYER_NUM;
        int j = rand() % LAYER_NEURON_NUM;
        if (i == 0)
        {                       // 第一层
            int k = rand() % 3; // 0代表参数x的weight,1代表参数y的weight,2代表参数bias
            if (k == 2)
            {
                while (1)
                {
                    double change = rand() % 2 ? 0.01 : -0.01;
                    neurons[0][j].bias = neurons[0][j].bias + change;
                    double r2 = check_data();
                    if (r < r2)
                    { // 误差变大了，改变测试方向。
                        change = 0 - change;
                        if (r2 - r < 0.01)
                        {
                            r = r2;
                            printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                            break;
                        }
                    }
                    else
                    {
                        if (r - r2 < 0.01)
                        {
                            r = r2;
                            printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                            break;
                        }
                    }
                    r = r2;
                    printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                }
            }
            else
            {
                while (1)
                {
                    double change = rand() % 2 ? 0.01 : -0.01;
                    neurons[0][j].weight[k] = neurons[0][j].weight[k] + change;
                    double r2 = check_data();
                    if (r < r2)
                    { // 误差变大了，改变测试方向。
                        change = 0 - change;
                        if (r2 - r < 0.01)
                        {
                            r = r2;
                            printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                            break;
                        }
                    }
                    else
                    {
                        if (r - r2 < 0.01)
                        {
                            r = r2;
                            printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                            break;
                        }
                    }
                    r = r2;
                    printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                }
            }
        }
        else if (i == HIDDEN_LAYER_NUM - 1)
        { // 最后一层
            int k = rand() % LAYER_NEURON_NUM;
            while (1)
            {
                double change = rand() % 2 ? 0.01 : -0.01;
                out_weight[k] = out_weight[k] + change;
                double r2 = check_data();
                if (r < r2)
                { // 误差变大了，改变测试方向。
                    change = 0 - change;
                    if (r2 - r < 0.01)
                    {
                        r = r2;
                        printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                        break;
                    }
                }
                else
                {
                    if (r - r2 < 0.01)
                    {
                        r = r2;
                        printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                        break;
                    }
                }
                r = r2;
                printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
            }
        }
        else
        { // 中间层
            int k = rand() % (LAYER_NEURON_NUM + 1);
            if (k == LAYER_NEURON_NUM)
            {
                while (1)
                {
                    double change = rand() % 2 ? 0.01 : -0.01;
                    neurons[i][j].bias = neurons[i][j].bias + change;
                    double r2 = check_data();
                    if (r < r2)
                    { // 误差变大了，改变测试方向。
                        change = 0 - change;
                        if (r2 - r < 0.01)
                        {
                            r = r2;
                            printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                            break;
                        }
                    }
                    else
                    {
                        if (r - r2 < 0.01)
                        {
                            r = r2;
                            printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                            break;
                        }
                    }
                    r = r2;
                    printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                }
            }
            else
            {
                while (1)
                {
                    double change = rand() % 2 ? 0.01 : -0.01;
                    neurons[i][j].weight[k] = neurons[i][j].weight[k] + change;
                    double r2 = check_data();
                    if (r < r2)
                    { // 误差变大了，改变测试方向。
                        change = 0 - change;
                        if (r2 - r < 0.0001)
                        {
                            r = r2;
                            printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                            break;
                        }
                    }
                    else
                    {
                        if (r - r2 < 0.0001)
                        {
                            r = r2;
                            printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                            break;
                        }
                    }
                    r = r2;
                    printf("i:%d,j:%d,k:%d,r:%f,at %d\n", i, j, k, r, __LINE__);
                }
            }
        }
    }
    printf("r:%f,at %d\n", r, __LINE__);
    return 0;
}

int main(int argc, char **argv)
{
    srand(time(0));
    init_neurons();
    init_trains_data();
    training();
    return 0;
}
