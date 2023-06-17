
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

__global__ void calcDispertion(int* a, int* b, int* reduced)
{
	__shared__ int cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Вычисление суммы разностей полученных результатов от ожидаемых
	for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
		cache[threadIdx.x] += (a[tid] - b[tid]) * (a[tid] - b[tid]);
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
	int a[N], b[N], reduced[blocksPerGrid];
	int* dev_a, * dev_b, * dev_reduced;

	// Выделение памяти
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_reduced, blocksPerGrid * sizeof(int));

	// Заполнение входных массивов
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i + i % 2;
	}

	// Передача данных устройству
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	// Исполнение на устройстве
	calcDispertion << < blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_reduced);

	// Передача данных хосту
	cudaMemcpy(reduced, dev_reduced, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

	// Вывод результирующих значений
	printf("\n Count of numbers: %d", N);
	printf("\n Blocks per grid: %d", blocksPerGrid);
	printf("\n Threads per block: %d", threadsPerBlock);
	printf("\n ---");
	
	float disp = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		printf("\n Temp sum of %d block is %d", i, reduced[i]);
		disp +=  reduced[i];
	}
	disp /= N;
	
	printf("\n ---");
	printf("\n Dispertion = %f\n", disp);

	// Освобождение памяти
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_reduced);

	return 0;
}
