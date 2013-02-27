#ifndef PARTICLE_H
#define PARTICLE_H

#define DIM 1
#define PARTICLE_COUNT 64000

typedef struct {
	float pos[DIM];
	float del[DIM];
	float bsf[DIM];
} particle;
typedef struct {
	particle s[PARTICLE_COUNT];
	particle d[PARTICLE_COUNT];
} blockData;

#endif
