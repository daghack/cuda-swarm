#define WINDOW_DIM 500
extern void (*cudaFunc)(float3 *);

void createVBO();
void initGLDevice();
void initGLUT();
void registerResources();
void drawParticles();
void display();
void initGraphics();

