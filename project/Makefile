CC = nvcc
HEADER_DIR = header/
PARTICLE_H = $(HEADER_DIR)particle.h
MAX_FUNC_H = $(HEADER_DIR)max_func.h

PSO_DIR = pso/
PSO = $(PSO_DIR)pso.o

WRAPPERS = wrapper_classes/
SWARM = $(WRAPPERS)Swarm.o

MAIN_DIR = main/
MAIN = $(MAIN_DIR)main.cu
OBJS = $(SWARM) $(PSO)
OUT = pso_test

all:
	make clean
	make main

pso.o :
	$(CC) -g -c -o $(PSO) $(PSO_DIR)pso.cu

Swarm.o : pso.o
	$(CC) -g -c -o $(SWARM) $(WRAPPERS)Swarm.cu

main : pso.o Swarm.o
	$(CC) -g -o $(OUT) $(OBJS) $(MAIN)

clean :
	\rm -f ./*/*.o $(OUT)
