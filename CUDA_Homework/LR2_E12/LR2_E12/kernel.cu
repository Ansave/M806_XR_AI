#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#define N 5120

using namespace std;

// Вычисление кол-ва пар
const int pairsCount = N / 6;
const int threadsPerBlock = 512;
const int blocksPerGrid = (pairsCount * 2 + threadsPerBlock - 1) / (threadsPerBlock);

__global__ void twinPrime(int * pairs)
{
	__shared__ int cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	cache[threadIdx.x] = tid / 2 * 6 + (tid % 2 ? 1 : -1);
	pairs[tid] = cache[threadIdx.x];

}

int main()
{
	// Инициализация массивов
	int pairs[pairsCount * 2 + 2], * dev_pairs;

	// Выделение памяти
	cudaMalloc((void**)&dev_pairs, (pairsCount * 2 + 2) * sizeof(int));

	// Передача данных устройству
	cudaMemcpy(dev_pairs, pairs, (pairsCount * 2 + 2) * sizeof(int), cudaMemcpyHostToDevice);

	// Исполнение на устройстве
	twinPrime << < blocksPerGrid, threadsPerBlock >> > (dev_pairs);

	// Передача данных хосту
	cudaMemcpy(pairs, dev_pairs, (pairsCount * 2 + 2) * sizeof(int), cudaMemcpyDeviceToHost);
	pairs[0] = 3;
	pairs[1] = 5;

	// Вывод результирующих значений
	printf("\n Count of numbers: %d", N);
	printf("\n Blocks per grid: %d", blocksPerGrid);
	printf("\n Threads per block: %d", threadsPerBlock);
	printf("\n ---");
	for (int i = 0; i < pairsCount * 2 + 2; i+=2)
	{
		printf("\n Pair (%d, %d)", pairs[i], pairs[i+1]);
	}
	printf("\n ---");

	// Освобождение памяти
	cudaFree(dev_pairs);

	return 0;
}