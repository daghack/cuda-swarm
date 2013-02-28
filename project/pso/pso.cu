#include <cuda.h>
#include <curand_kernel.h>
#include "pso.h"
#include "../headers/max_func.h"

#define W 1.0
#define C1 1.0
#define C2 1.0

__device__ void src_to_dest(particle * s, particle * d, unsigned int index, unsigned int dim) {
	d[index].pos[dim] = s[index].pos[dim];
	d[index].del[dim] = 0.0;
	d[index].bsf[dim] = s[index].bsf[dim];
}

__device__ float * global(particle * s, unsigned int i) {
	unsigned int pu, pd;
	float * a, * b, * c;
	float am, bm, cm;

	pu = (i + 1) % DIM;
	pd = (i - 1) % DIM;
	a = s[i].bsf;
	am = max_func(a);
	b = s[pu].bsf;
	bm = max_func(b);
	c = s[pd].bsf;
	cm = max_func(c);
	
	return (am > bm) ? (am > cm ? a : c) : (bm > cm ? b : c);
}

__device__ float inertial(float del) {
	return W * del;
}

__device__ float cognitive(float pos, float bsf) {
	return C1 * (bsf - pos);
}

__device__ float social(particle * s, particle * d, unsigned int i, unsigned int dim) {
	float * k = global(s, i);
	return C2 * (k[dim] - s[i].pos[dim]);
}

__device__ void update_best(particle * s, particle * d, unsigned int index) {
	if (max_func(s[index].pos) > max_func(s[index].bsf)) {
		for(unsigned int i = 0; i < DIM; i++) {
			d[index].bsf[i] = s[index].pos[i];
		}
	}
}

__device__ void update(particle * s, particle * d, curandState_t * state, unsigned int index) {
	for(unsigned int i = 0; i < DIM; i++) {
		src_to_dest(s, d, index, i);
		d[index].del[i] += inertial(s[index].del[i]);
		d[index].del[i] += cognitive(s[index].pos[i], s[index].bsf[i]) * curand_uniform(&state[index]);
		d[index].del[i] += social(s, d, index, i) * curand_uniform(&state[index]);
		d[index].pos[i] += d[index].del[i];
	}
}

__global__ void initBlock(blockData * p, unsigned int seed, float min, float max) {
	unsigned int x_i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y_i = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned i = x_i + y_i * blockDim.x * gridDim.x;
	
	if (i < PARTICLE_COUNT) {
		curand_init(seed, i, 0, &(p->states[i]));
		for (unsigned int j = 0; j < DIM; j++) {
			float k = min + (max - min) * curand_uniform(&(p->states[i]));
			p->s[i].pos[j] = k;
			p->s[i].bsf[j] = k;
		}
	}
}

__global__ void pso(blockData * p, bool sw) {
	unsigned int x_i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y_i = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned i = x_i + y_i * blockDim.x * gridDim.x;
	
	particle * s = sw ? (particle *)p->s : (particle *)p->d;
	particle * d = sw ? (particle *)p->d : (particle *)p->s;
	curandState_t * states = (curandState_t *)p->states;
	
	if (i < PARTICLE_COUNT) {
		update(s, d, states, i);
		update_best(s, d, i);
	}
}
