#include <cuda.h>
#include <stdio.h>
#include "../pso/pso.h"

dim3 threadGrid(1, 1, 1);
dim3 threadBlock(64, 64, 64);

bool swapper = true;
blockData * hostData = NULL;
blockData * deviData = NULL;
unsigned int blockSize = sizeof(blockData);

void initializeData() {
	hostData = (blockData *)malloc(blockSize);
	memset(hostData, 0, blockSize);
	cudaMalloc((void **)&deviData, blockSize);
	cudaMemcpy(hostData, deviData, blockSize, cudaMemcpyHostToDevice);
}

void runPSO(unsigned int iterations) {
	for(unsigned int i = 0; i < iterations; i++) {
		pso<<<threadGrid, threadBlock>>>(deviData, swapper);
		swapper = !swapper;
	}
}

int main(int argc, char ** argv) {
	printf("blockDataSize : %d\n", blockSize);
	initializeData();
	runPSO(1000);
	return 0;
}
