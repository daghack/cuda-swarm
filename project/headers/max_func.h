#include <cuda.h>

// INVERSE DISTANCE FROM (0, 0, ...) //
__device__ float max_func(float * s) {
	float n = 0.0;
	for(unsigned int i = 0; i < DIM; i++) {
		n += s[i] * s[i];
	}
	return (1.0/(sqrtf(n)+1)) * 100.0;
	
}

__device__ float neg_cost_squared(float * s, float2 * p, unsigned int pcount) {
	float cost = 0;
	for(unsigned int point = 0; point < pcount; point++) {
		float scalar = 1;
		float val = 0;
		for(unsigned int i = 0; i < DIM; i++) {
			val += scalar * s[i];
			scalar *= p[point].x;
		}
		float toAdd = p[point].y - val;
		cost += toAdd * toAdd;
	}
	return (-0.5 / pcount) * cost;
}
