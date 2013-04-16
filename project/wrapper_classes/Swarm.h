#include <cuda.h>
#include "../headers/particle.h"

typedef float(*func)(float *, unsigned int n);

class Swarm {
	private:
		blockData * hostData;
		blockData * deviData;
		dim3 threadGrids, threadBlocks;
		bool swapper;
		
		void init();
		void del();
		void copyDeviceToHost();
	public:
		Swarm();
		Swarm(unsigned int, float, float, float, float, float);
		~Swarm();
		void restartSwarm();
		void restartSwarm(unsigned int, float, float, float, float, float);
		void runIteration();
		void runNIterations(unsigned int);
		void printCurrent();
};
