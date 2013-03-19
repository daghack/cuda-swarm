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
const float MINVEL = -10.0;
const float MAXVEL = 10.0;

void initialize() {
	hostData = (blockData *)malloc(blockSize);
	memset(hostData, 0, blockSize);
	cudaMalloc((void **)&deviData, blockSize);
	//CudaMemcpy(void * dest, void * host, size, direction)
	cudaMemcpy(deviData, hostData, blockSize, cudaMemcpyHostToDevice);
	initBlock<<<threadGrids, threadBlocks>>>(deviData, SEED, 10.0, MININIT, MAXINIT, MINVEL, MAXVEL);
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
		printf("<particle Source %d>\n", i);
		printf("<particle Destination %d>\n", i);
		for(unsigned int j = 0; j < DIM; j++) {
			printf("\t<dim Source %d: %.2f>\n", j, hostData->s[i].pos[j]);
			printf("\t<dim Destination %d: %.2f>\n", j, hostData->d[i].pos[j]);
			printf("\t<bestDim Source %d: %.2f>\n", j, hostData->s[i].bsf[j]);
			printf("\t<bestDim Destination%d: %.2f>\n", j, hostData->d[i].bsf[j]);
			printf("\t<delta Source %d: %.2f>\n", j, hostData->s[i].del[j]);
			printf("\t<delta Destination %d: %.2f>\n", j, hostData->d[i].del[j]);
		}
	}
}

int main(int argc, char ** argv) {
	unsigned int iterations = 0;
	if (argc != 2 || atoi(argv[1]) == 0) {
		printf("Usage : %s <Number of Iterations>\n", argv[0]);
		exit(0);
	}
	else {
		iterations = atoi(argv[1]);
	}
	printf("blockDataSize : %d\n", blockSize);
	initialize();
	printOutHostData();
	runPSO(iterations);
	copyResultsBack();
	printOutHostData();
	finalize();
	return 0;
}
