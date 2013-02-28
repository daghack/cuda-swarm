#include <cuda.h>

// INVERSE DISTANCE FROM (0, 0, ...) //
__device__ float max_func(float * s) {
	float n = 0.0;
	for(unsigned int i = 0; i < DIM; i++) {
		n += s[i] * s[i];
	}
	if (n == 0) {
		return 1000.0;
	}
	return 1.0/sqrtf(n) * 100.0;
	
}
