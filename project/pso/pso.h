#define W 1.0  //INERTIAL
#define C1 1.2 //COGNITIVE
#define C2 1.2 //SOCIAL

#include "../headers/particle.h"

typedef float (*map)(float n);
typedef float (*fold)(float * n);

__global__ void initBlock(blockData *, unsigned int, float, float, float, float, float);
//--------------------------------------------------------------------------------------------------------//
//USING DEFINED MAX_FUNC
//--------------------------------------------------------------------------------------------------------//
__device__ void src_to_dest(particle *, particle *, unsigned int, unsigned int);
__device__ float global(particle *, unsigned int, unsigned int);
__device__ float inertial(float);
__device__ float cognitive(float, float);
__device__ float social(particle *, particle *, unsigned int);
__device__ void update(particle *, particle *, unsigned int);
__device__ void update_best(particle *, particle *, unsigned int);
__global__ void pso(blockData *, bool);

//--------------------------------------------------------------------------------------------------------//
//USING MAP MAX_FUNC
//--------------------------------------------------------------------------------------------------------//
__device__ unsigned int global(particle *, unsigned int, map);
__device__ void update_best(particle *, particle *, curandState_t *, unsigned int, map *);
__global__ void pso(blockData *, bool, map);
__device__ float map_wrapper(float *, map);

//--------------------------------------------------------------------------------------------------------//
//USING FOLD MAX_FUNC
//--------------------------------------------------------------------------------------------------------//
__device__ unsigned int global(particle *, unsigned int, fold);
__device__ void update_best(particle *, particle *, curandState_t *, unsigned int, fold);
__global__ void pso(blockData *, bool, fold);

