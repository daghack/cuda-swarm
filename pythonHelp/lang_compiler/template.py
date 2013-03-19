def generateCode(name, body):
	return "#include <cuda.h>\n__device__ float %s(float * s) {%s}" % (name, body)
