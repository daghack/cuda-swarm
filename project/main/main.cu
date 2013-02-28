#include <cuda.h>
#include <stdio.h>
#include "../pso/pso.h"

dim3 threadGrids(128, 128, 1); // MAX NUMBER OF BLOCKS : 65535 ^ 3
dim3 threadBlocks(512, 1, 1); // MAX NUMBER OF THREADS : 512

bool swapper = true;
blockData * hostData = NULL;
blockData * deviData = NULL;
unsigned int blockSize = sizeof(blockData);

const unsigned int SEED = 42;
const float MININIT = -100.0;
const float MAXINIT = 100.0;

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
	//CudaMemcpy(void * dest, void * host, size, direction)
	cudaMemcpy(deviData, hostData, blockSize, cudaMemcpyHostToDevice);
	initBlock<<<threadGrids, threadBlocks>>>(deviData, SEED, MININIT, MAXINIT);
	cudaMemcpy(hostData, deviData, blockSize, cudaMemcpyDeviceToHost);
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

void printOutHostData() {
	for(unsigned int i = 0; i < PARTICLE_COUNT; i++) {
		printf("<particle %d>\n", i);
		for(unsigned int j = 0; j < DIM; j++) {
			printf("\t<dim %d: %.2f>\n", j, hostData->s[i].pos[j]);
			printf("\t<bestDim %d: %.2f>\n", j, hostData->s[i].bsf[j]);
		}
	}
}

int main(int argc, char ** argv) {
	printf("blockDataSize : %d\n", blockSize);
	initialize();
	printOutHostData();
	runPSO(1);
	copyResultsBack();
	printOutHostData();
	finalize();
	return 0;
}
