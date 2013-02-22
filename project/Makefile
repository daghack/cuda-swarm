CC = nvcc
HEADER_DIR = header/
PARTICLE_H = $(HEADER_DIR)particle.h
MAX_FUNC_H = $(HEADER_DIR)max_func.h

PSO_DIR = pso/
MAX_FUNC = $(MAX_FUNC_DIR)max_func.o
PSO = $(PSO_DIR)pso.o

MAIN_DIR = main/
MAIN = $(MAIN_DIR)main.cu
OBJS = $(MAX_FUNC) $(PSO)
OUT = pso_test

all:
	make main

pso.o :
	$(CC) -c -o $(PSO) $(PSO_DIR)pso.cu

main : pso.o
	$(CC) -o $(OUT) $(PSO) $(MAIN)

clean :
	\rm ./*/*.o $(OUT)
