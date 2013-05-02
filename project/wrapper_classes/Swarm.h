#include <cuda.h>
#include "../headers/particle.h"
#include "../pso/pso.h"

typedef float(*func)(float *, unsigned int n);


template <typename func>
class Swarm {
	private:
		blockData * hostData;
		blockData * deviData;
		func f;
		dim3 threadGrids, threadBlocks;
		bool swapper;
		
		void init();
		void del();
		void copyDeviceToHost();
	public:
		Swarm();
		Swarm(unsigned int, float, float, float, float, float, func);
		~Swarm();
		void restartSwarm();
		void restartSwarm(unsigned int, float, float, float, float, float, func);
		void runIteration();
		void runNIterations(unsigned int);
		void printCurrent();
};
