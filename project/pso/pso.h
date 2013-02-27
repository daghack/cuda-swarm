#define W 1.0
#define C1 1.0
#define C2 1.0

#include "../headers/particle.h"

__device__ void src_to_dest(particle *, particle *, unsigned int, unsigned int);
__device__ float global(particle *, unsigned int, unsigned int);
__device__ float inertial(float);
__device__ float cognitive(float, float);
__device__ float social(particle *, particle *, unsigned int);
__device__ void update(particle *, particle *, unsigned int);
__device__ void update_best(particle *, particle *, unsigned int);

__global__ void initBlock(blockData *, unsigned int, float, float);
__global__ void pso(blockData *, bool);
