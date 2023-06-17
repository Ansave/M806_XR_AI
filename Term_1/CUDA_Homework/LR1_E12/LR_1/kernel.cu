
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

#define N 1024
//#define N 2048
//#define N 4096
//#define N 8192
//#define N 16384
//#define N 32768
//#define N 65536

__global__ void square(int * in, int * out)
{
	out[blockIdx.x] = in[blockIdx.x] * in[blockIdx.x];
}

void squareCPU(int* in, int* out) 
{
	for (int i = 0; i < N; i++)
	{
		out[i] = in[i] * in[i];
	}
}

int main()
{
	// Инициализация массивов
	int in[N], out[N];
	int* dev_in, * dev_out;

	// Выделение памяти
	cudaMalloc((void**)&dev_in, N * sizeof(int));
	cudaMalloc((void**)&dev_out, N * sizeof(int));

	// Заполнение входного массива
	for (int i = 0; i < N; i++)
	{
		in[i] = i;
	}

	// Передача данных устройству
	cudaMemcpy(dev_in, in, N * sizeof(N), cudaMemcpyHostToDevice);

	// Иницализация переменных мониторинга производительности
	double executionTime;
	int iterations = 10;

	// Исполнение на устройстве
	clock_t startGPU = clock();
	for (int i = 0; i < iterations; i++)
	{
		square <<<N, 1 >>> (dev_in, dev_out);
	}
	executionTime = ((double)clock() - startGPU) / CLOCKS_PER_SEC;
	printf("\n GPU Execution time is %.60lf", executionTime);

	// Исполнение на хосте
	clock_t startСPU = clock();
	for (int i = 0; i < iterations; i++)
	{
		squareCPU(in, out);
	}
	executionTime = ((double)clock() - startСPU) / CLOCKS_PER_SEC;
	printf("\n CPU Execution time is %.60lf", executionTime);

	// Передача данных хосту
	cudaMemcpy(out, dev_out, N * sizeof(N), cudaMemcpyDeviceToHost);

	// Вывод результирующих значений
	for (int i = 0; i < N; i++)
	{
		printf("\n Square of %d is %d", i, out[i]);
	}

	// Освобождение памяти
	cudaFree(dev_in);
	cudaFree(dev_out);

	return 0;
}
