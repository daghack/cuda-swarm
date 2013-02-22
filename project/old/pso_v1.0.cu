#include <cuda.h>
#include <stdio.h>

#define DIM 1
#define PARTICLE_COUNT 512

#define W 2.0
#define C1 1.0
#define C2 1.0

typedef struct {
	float pos[DIM];
	float del[DIM];
	float bsf[DIM];
} particle;

typedef struct {
	particle s[PARTICLE_COUNT];
	particle d[PARTICLE_COUNT];
} blockData;

__device__ void init(particle * s, particle * d, unsigned int index) {
	for(unsigned int i = 0; i < DIM; i++) {
		d[index].pos[i] = s[index].pos[i];
		d[index].del[i] = s[index].del[i];
		d[index].bsf[i] = s[index].bsf[i];
	}
}

__device__ float global(particle * s, unsigned int i, unsigned int d) {
	unsigned int pu, pd;
	pu = (i + 1) % DIM;
	pd = (i - 1) % DIM;
	float a = s[i].bsf[d];
	float b = s[pu].bsf[d];
	float c = s[pd].bsf[d];
	return (a > b) ? (a > c ? a : c) : ( b > c ? b : c);

}

__device__ void inertial(particle * s, particle * d, unsigned int index) {
	for(unsigned int i = 0; i < DIM; i++) {
		d[index].del[i] += s[index].del[i] * W;
	}
}

__device__ void cognitive(particle * s, particle * d, unsigned int index) {
	for(unsigned int i = 0; i < DIM; i++) {
		d[index].del[i] += C1 * (s[index].bsf[i] - s[index].pos[i]);
	}
}

__device__ void social(particle * s, particle * d, unsigned int index) {
	for(unsigned int i = 0; i < DIM; i++) {
		d[index].del[i] += C2 * (global(s, index, i) - s[index].pos[i]);
	}
	//GLOBAL BEST
}

__global__ void pso(blockData * p, float w, float c1, float c2, bool sw) {
	unsigned int x_i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y_i = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned index = x_i + y_i * blockDim.x * gridDim.x;
	
	particle * s = sw ? (particle *)p->s : (particle *)p->d;
	particle * d = sw ? (particle *)p->d : (particle *)p->s;
	
	if (index < PARTICLE_COUNT) {
		init(s, d, index);
		inertial(s, d, index);
		cognitive(s, d, index);
		social(s, d, index);
	}
}

int main() {
	return 0;
}
