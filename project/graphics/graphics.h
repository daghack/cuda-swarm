#define WINDOW_DIM 500
extern void (*cudaFunc)();

void createVBO();
void initGLDevice();
void initGLUT();
void registerResources();
void drawParticles();
void display();
void initGraphics();

