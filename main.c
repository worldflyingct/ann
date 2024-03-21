#include <stdio.h>
#include <string.h>

#define LAYER_NEURON_NUM   8
#define HIDDEN_LAYER_NUM   8

struct NEURON
{
    int value;
    int bias;
    float weight[HIDDEN_LAYER_NUM];
};
struct NEURON neurons[HIDDEN_LAYER_NUM][LAYER_NEURON_NUM];
int out_weight[LAYER_NEURON_NUM]; // 最后一级的权重

int calc(int x, int y)
{
    for (int i = 0 ; i < LAYER_NEURON_NUM ; i++)
    {
        int sum = 0;
        sum += x * neurons[0][i].weight[0];
        sum += y * neurons[0][i].weight[1];
        neurons[0][i].value = sum + neurons[0][i].bias;
    }
    for (int i = 1 ; i < HIDDEN_LAYER_NUM ; i++)
    {
        for (int j = 0 ; j < LAYER_NEURON_NUM ; j++)
        {
            int sum = 0;
            for (int k = 0 ; k < LAYER_NEURON_NUM ; k++)
            {
                sum += neurons[i-1][k].value * neurons[i][j].weight[k];
            }
            neurons[i][j].value = sum + neurons[i][j].bias;
        }
    }
    int out = 0;
    for (int i = 0 ; i < LAYER_NEURON_NUM ; i++)
    {
        out += neurons[HIDDEN_LAYER_NUM-1][i].value * out_weight[i];
    }
    return out;
}

void init_neurons()
{
    for (int i = 0 ; i < HIDDEN_LAYER_NUM ; i++)
    {
        for (int j = 0 ; j < LAYER_NEURON_NUM ; j++)
        {
            neurons[i][j].bias = 0;
            neurons[i][j].value = 0;
            for (int k = 0 ; k < LAYER_NEURON_NUM ; k++)
            {
                neurons[i][j].weight[k] = 1;
            }
            out_weight[j] = 1;
        }
    }
}

int trains_data[50][50];
void init_trains_data()
{
    memset(trains_data, 0, sizeof(trains_data));
    trains_data[11][22] = 1;
    trains_data[10][37] = 1;
    trains_data[26][15] = 1;
    trains_data[5][7] = 1;
    trains_data[8][49] = 1;
    trains_data[24][43] = -1;
    trains_data[27][22] = -1;
    trains_data[44][12] = -1;
    trains_data[32][34] = -1;
    trains_data[34][36] = -1;
    trains_data[23][24] = -1;
    trains_data[23][26] = -1;
}

int check_data()
{
    int right = 0;
    int total = 0;
    for (int x = 0 ; x < 50 ; x++)
    {
        for (int y = 0 ; y < 50 ; y++)
        {
            if (trains_data[x][y] != 0)
            {
				total++;
                int r = calc(x, y);
                if ((r > 0 && trains_data[x][y] > 0) || (r < 0 && trains_data[x][y] < 0))
                {
                    right++;
                }
            }
        }
    }
    return 100 * right / total;
}

int training ()
{
	
}

int main(int argc, char **argv)
{
    init_neurons();
    init_trains_data();
    int r = check_data();
    printf("r:%d\n", r);
    return 0;
}

