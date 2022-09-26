#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

//#define N 1024
//#define N 2048
//#define N 4096
#define N 8192
//#define N 16384
//#define N 32768
//#define N 65536

using namespace std;

#define imin(a,b) (a<b?a:b)

const int threadsPerBlock = 512;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void reduction( int* in, unsigned long long int* reduced)
{
	__shared__ unsigned long long int cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Сложение значений входного массива
	for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
		cache[threadIdx.x] += in[tid] * in[tid];
	}

	__syncthreads();

	// Непосредственно редукция
	for (int i = blockDim.x / 2; (i != 0) && (threadIdx.x < i); i /= 2) {
		cache[threadIdx.x] += cache[threadIdx.x + i];
		__syncthreads();
	}

	// Запись частичных сумм в массив меньшего размера (Результат выполнения редукции)
	if (threadIdx.x == 0) {
		reduced[blockIdx.x] = cache[0];
	}
}

int main()
{
	// Инициализация массивов
	int in[N], * dev_in;
	unsigned long long int reduced[blocksPerGrid], * dev_reduced;

	// Выделение памяти
	cudaMalloc((void**)&dev_in, N * sizeof(int));
	cudaMalloc((void**)&dev_reduced, blocksPerGrid * sizeof(long long int));

	// Заполнение входного массива (арифметическая прогрессия)
	for (int i = 0; i < N; i++)
	{
		in[i] = i;
	}

	// Передача данных устройству
	cudaMemcpy(dev_in, in, N * sizeof(int), cudaMemcpyHostToDevice);

	// Исполнение на устройстве
	reduction << < blocksPerGrid, threadsPerBlock >> > (dev_in, dev_reduced);

	// Передача данных хосту
	cudaMemcpy(reduced, dev_reduced, blocksPerGrid * sizeof(long long int), cudaMemcpyDeviceToHost);

	// Вывод результирующих значений
	printf("\n Count of numbers: %d", N);
	printf("\n Blocks per grid: %d", blocksPerGrid);
	printf("\n Threads per block: %d", threadsPerBlock);
	printf("\n ---");
	unsigned long long int sum = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		printf("\n Temp sum of %d block is %lld", i, reduced[i]);
		sum += reduced[i];
	}
	printf("\n ---");
	printf("\n Result sum = %lld\n", sum);

	// Освобождение памяти
	cudaFree(dev_in);
	cudaFree(dev_reduced);

	return 0;
}