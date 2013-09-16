#include <stdio.h>
#include "Swarm.h"

//-------------------------------------------------------------//

template <typename func>
void Swarm<func>::init() {
	//The swapper is the boolean which indicates which "frame" for the graphics kernal to use.
	swapper = 1;
	threadGrids = dim3(128, 128, 1);
	threadBlocks = dim3(512, 1, 1);
	//Allocate memory for host data and device data
	this->hostData = (blockData *)malloc(sizeof(blockData));
	cudaMalloc((void **)&deviData, sizeof(blockData));;
	
	//Clear out both the host data and device data
	memset(hostData, 0, sizeof(blockData));
	cudaMemcpy(deviData, hostData, sizeof(blockData), cudaMemcpyHostToDevice);
}
template <typename func>
void Swarm<func>::del() {
	free(hostData);
	cudaFree(deviData);
}
template <typename func>
void Swarm<func>::copyDeviceToHost() {
	cudaMemcpy(hostData, deviData, sizeof(blockData), cudaMemcpyDeviceToHost);
}

//-------------------------------------------------------------//
template <typename func>
Swarm<func>::Swarm() {
	init();
	initBlock<<<threadGrids, threadBlocks>>>(deviData, 0, 10.0, -100.0, 100.0, -10.0, 10.0);
}
template <typename func>
Swarm<func>::Swarm(unsigned int seed, float maxVelocity, float minPosition, float maxPosition, float minDelta, float maxDelta, func f) {
	init();
	initBlock<<<threadGrids, threadBlocks>>>(deviData, seed, maxVelocity, minPosition, maxPosition, minDelta, maxDelta);
	cudaMemcpyFromSymbol(&(this->f), f, sizeof(func));
}
template <typename func>
Swarm<func>::~Swarm() {
	del();
}
template <typename func>
void Swarm<func>::restartSwarm() {
	del();
	init();
	initBlock<<<threadGrids, threadBlocks>>>(deviData, 0, 10.0, -100.0, 100.0, -10.0, 10.0);
}
template <typename func>
void Swarm<func>::restartSwarm(unsigned int s, float mV, float minP, float maxP, float minD, float maxD, func f) {
	del();
	init();
	initBlock<<<threadGrids, threadBlocks>>>(deviData, s, mV, minP, maxP, minD, maxD);
	cudaMemcpyFromSymbol(&(this->f), f, sizeof(func));
}
template<>
void Swarm<map>::runIteration() {
	pso<<<threadGrids, threadBlocks>>>(deviData, swapper, f);
	swapper = !swapper;
}
template<>
void Swarm<fold>::runIteration() {
	pso<<<threadGrids, threadBlocks>>>(deviData, swapper, f);
	swapper = !swapper;
}
template <typename func>
void Swarm<func>::runIteration() {
	printf("NOT VALID TEMPLATE\n");
}
template <typename func>
void Swarm<func>::runNIterations(unsigned int n) {
	for(unsigned int i = 0; i < n; i++) {
		runIteration();
	}
}
template <typename func>
void Swarm<func>::printCurrent() {
	copyDeviceToHost();
	for(unsigned int i = 0; i < PARTICLE_COUNT; i++) {
		printf("<Particle: %d>\n", i);
		for(unsigned int j = 0; j < DIM; j++) {
			printf("\t<FrameA>\n");
			printf("\t\t<dim %d: %.2f>\n", j, hostData->s[i].pos[j]);
			printf("\t\t<bestDim %d: %.2f>\n", j, hostData->s[i].bsf[j]);
			printf("\t\t<delta %d: %.2f>\n", j, hostData->s[i].del[j]);
			printf("\t<FrameB>\n");
			printf("\t\t<dim %d: %.2f>\n", j, hostData->d[i].pos[j]);
			printf("\t\t<bestDim %d: %.2f>\n", j, hostData->d[i].bsf[j]);
			printf("\t\t<delta %d: %.2f>\n", j, hostData->d[i].del[j]);
		}
	}
}

template class Swarm<fold>;
template class Swarm<map>;
