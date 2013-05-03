#include <cuda.h>
#include <curand_kernel.h>
#include "pso.h"
#include "../headers/max_func.h"

__device__ float map_wrapper(float * s, map f) {
	float toret = 0;
	for(unsigned int i = 0; i < DIM; i++) {
		toret += f(s[i]);
	}
	return toret;
}

__device__ void src_to_dest(particle * s, particle * d, unsigned int index, unsigned int dim) {
	d[index].pos[dim] = s[index].pos[dim];
	d[index].del[dim] = 0.0;
	d[index].bsf[dim] = s[index].bsf[dim];
	d[index].max_v = s[index].max_v;
}
__device__ unsigned int global(particle * s, unsigned int i) {
	unsigned int pu, pd;
	float am, bm, cm;

	pu = (i + 1) % DIM;
	pd = (i - 1) % DIM;
	am = max_func(s[i].bsf);
	bm = max_func(s[pu].bsf);
	cm = max_func(s[pd].bsf);
	
	if (am > bm) {
		if (am > cm) {
			return i;
		}
		else {
			return pd;
		}
	}
	else {
		if (bm > cm) {
			return pu;
		}
		else {
			return pd;
		}
	}
}
__device__ unsigned int global(particle * s, unsigned int i, map f) {
	unsigned int pu, pd;
	float am, bm, cm;

	pu = (i + 1) % DIM;
	pd = (i - 1) % DIM;
	am = map_wrapper(s[i].bsf, f);
	bm = map_wrapper(s[pu].bsf, f);
	cm = map_wrapper(s[pd].bsf, f);
	
	if (am > bm) {
		if (am > cm) {
			return i;
		}
		else {
			return pd;
		}
	}
	else {
		if (bm > cm) {
			return pu;
		}
		else {
			return pd;
		}
	}
}
__device__ unsigned int global(particle * s, unsigned int i, fold f) {
	unsigned int pu, pd;
	float am, bm, cm;

	pu = (i + 1) % DIM;
	pd = (i - 1) % DIM;
	am = f(s[i].bsf);
	bm = f(s[pu].bsf);
	cm = f(s[pd].bsf);
	
	if (am > bm) {
		if (am > cm) {
			return i;
		}
		else {
			return pd;
		}
	}
	else {
		if (bm > cm) {
			return pu;
		}
		else {
			return pd;
		}
	}
}

__device__ float inertial(float del) {
	return W * del;
}

__device__ float cognitive(float pos, float bsf) {
	return C1 * (bsf - pos);
}

__device__ float social(particle * s, particle * d, unsigned int i, unsigned int dim) {
	unsigned int best_index = global(s, i);
	return C2 * (s[best_index].bsf[dim] - s[i].pos[dim]);
}

__device__ void update_best(particle * s, particle * d, unsigned int index) {
	if (max_func(s[index].pos) > max_func(s[index].bsf)) {
		for(unsigned int i = 0; i < DIM; i++) {
			d[index].bsf[i] = s[index].pos[i];
		}
	}
}
__device__ void update_best(particle * s, particle * d, unsigned int index, map f) {
	if (map_wrapper((s[index].pos), f) > map_wrapper(s[index].bsf, f)) {
		for(unsigned int i = 0; i < DIM; i++) {
			d[index].bsf[i] = s[index].pos[i];
		}
	}
}
__device__ void update_best(particle * s, particle * d, unsigned int index, fold f) {
	if (f(s[index].pos) > f(s[index].bsf)) {
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
		
		float delp = d[index].del[i];
		if(abs(d[index].del[i]) > delp) {
			delp = d[index].max_v;
			if(d[index].del[i] < 0) {
				delp *= -1;
			}
			d[index].del[i] = delp;
		}
		
		d[index].pos[i] += delp;

	}
}

__global__ void initBlock(blockData * p, unsigned int seed, float max_v, float pos_min, float pos_max, float del_min, float del_max) {
	unsigned int x_i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y_i = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned i = x_i + y_i * blockDim.x * gridDim.x;
	
	if (i < PARTICLE_COUNT) {
		curand_init(seed, i, 0, &(p->states[i]));
		for (unsigned int j = 0; j < DIM; j++) {
			float k = (pos_max - pos_min) * curand_uniform(&(p->states[i])) + pos_min;
			p->s[i].pos[j] = k;
			p->s[i].bsf[j] = k;
			p->s[i].del[j] = (del_max - del_min) * curand_uniform(&(p->states[i])) + del_min;
			p->s[i].max_v = max_v;
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
__global__ void pso(blockData * p, bool sw, fold f) {
	unsigned int x_i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y_i = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned i = x_i + y_i * blockDim.x * gridDim.x;
	
	particle * s = sw ? (particle *)p->s : (particle *)p->d;
	particle * d = sw ? (particle *)p->d : (particle *)p->s;
	curandState_t * states = (curandState_t *)p->states;
	
	if (i < PARTICLE_COUNT) {
		update(s, d, states, i);
		update_best(s, d, i, f);
	}
}
__global__ void pso(blockData * p, bool sw, map f) {
	unsigned int x_i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y_i = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned i = x_i + y_i * blockDim.x * gridDim.x;
	
	particle * s = sw ? (particle *)p->s : (particle *)p->d;
	particle * d = sw ? (particle *)p->d : (particle *)p->s;
	curandState_t * states = (curandState_t *)p->states;
	
	if (i < PARTICLE_COUNT) {
		update(s, d, states, i);
		update_best(s, d, i, f);
	}
}
////////////////////////////////////////////////////////////////////////////////////////////
__device__ float linear_regression_cost(float * s, float2 * f, unsigned int c) {
	float cost = 0.0;
	for(unsigned int p = 0; p < c; p++) {
		float y_prime = 0.0;
		float x_prime = 1.0;
		for(unsigned int i = 0; i < DIM; i++) {
			y_prime += x_prime * s[i];
			x_prime *= f[p].x;
		}
		float t_c = y_prime - f[p].y;
		cost += t_c * t_c;
	}
	return cost/(2 * c);
}

__device__ void update_best(particle * s, particle * d, unsigned int index, float2 * f, unsigned int c) {
	if (linear_regression_cost(s[index].pos, f, c) > linear_regression_cost(s[index].bsf, f, c)) {
		for(unsigned int i = 0; i < DIM; i++) {
			d[index].bsf[i] = s[index].pos[i];
		}
	}
}
__device__ unsigned int global(particle * s, unsigned int i, float2 * f, unsigned int c) {
	unsigned int pu, pd;
	float am, bm, cm;

	pu = (i + 1) % DIM;
	pd = (i - 1) % DIM;
	am = linear_regression_cost(s[i].bsf, f, c);
	bm = linear_regression_cost(s[pu].bsf, f, c);
	cm = linear_regression_cost(s[pd].bsf, f, c);
	
	if (am > bm) {
		if (am > cm) {
			return i;
		}
		else {
			return pd;
		}
	}
	else {
		if (bm > cm) {
			return pu;
		}
		else {
			return pd;
		}
	}
}
__global__ void pso(blockData *p, bool sw, float2 * f, unsigned int c) {
	unsigned int x_i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y_i = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned i = x_i + y_i * blockDim.x * gridDim.x;
	
	particle * s = sw ? (particle *)p->s : (particle *)p->d;
	particle * d = sw ? (particle *)p->d : (particle *)p->s;
	curandState_t * states = (curandState_t *)p->states;
	
	if (i < PARTICLE_COUNT) {
		update(s, d, states, i);
		update_best(s, d, i, f, c);
	}
}
