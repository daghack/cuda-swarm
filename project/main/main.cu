#include <cuda.h>
#include <stdio.h>
#include "../pso/pso.h"

dim3 threadGrids(1, 1, 1);
dim3 threadBlocks(64, 64, 64);

bool swapper = true;
blockData * hostData = NULL;
blockData * deviData = NULL;
unsigned int blockSize = sizeof(blockData);

unsigned int SEED = 42;

__global__ void init_blockData(blockData * s, unsigned int seed) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int i = x + y * blockDim.x * gridDim.x;
	
	if (i < PARTICLE_COUNT) {
		curand_init(seed, i, 0, &(s->states[i]));
	}
	
}

void initialize() {
	hostData = (blockData *)malloc(blockSize);
	memset(hostData, 0, blockSize);
	cudaMalloc((void **)&deviData, blockSize);
	cudaMemcpy(hostData, deviData, blockSize, cudaMemcpyHostToDevice);
	init_blockData<<<threadGrids, threadBlocks>>>(deviData, SEED);
}

void finalize() {
	free(hostData);
	cudaFree(deviData);
}

void runPSO(unsigned int iterations) {
	for(unsigned int i = 0; i < iterations; i++) {
		pso<<<threadGrids, threadBlocks>>>(deviData, swapper);
		swapper = !swapper;
	}
}

void copyResultsBack() {
	cudaMemcpy(hostData, deviData, blockSize, cudaMemcpyDeviceToHost);
}

int main(int argc, char ** argv) {
	printf("blockDataSize : %d\n", blockSize);
	initialize();
	runPSO(1000);
	
	finalize();
	return 0;
}
