CC = nvcc
HEADER_DIR = header/
PARTICLE_H = $(HEADER_DIR)particle.h
MAX_FUNC_H = $(HEADER_DIR)max_func.h

PSO_DIR = pso/
MAX_FUNC = $(MAX_FUNC_DIR)max_func.o
PSO = $(PSO_DIR)pso.o

GRAPHICS_DIR = graphics/
GRAPHICS = $(GRAPHICS_DIR)graphics.o

MAIN_DIR = main/
MAIN = $(MAIN_DIR)main.cu
OBJS = $(MAX_FUNC) $(PSO) $(GRAPHICS)
OUT = pso_test

EXTERNAL = -lGLEW -lglut -lGLU -lGL

all:
	make clean
	make main

graphics.o :
	$(CC) -g -c -o $(GRAPHICS) $(GRAPHICS_DIR)graphics.cu

pso.o :
	$(CC) -g -c -o $(PSO) $(PSO_DIR)pso.cu

main : pso.o graphics.o
	$(CC) -g -o $(OUT) $(PSO) $(MAIN) $(GRAPHICS) $(EXTERNAL)

clean :
	\rm -f ./*/*.o $(OUT)
