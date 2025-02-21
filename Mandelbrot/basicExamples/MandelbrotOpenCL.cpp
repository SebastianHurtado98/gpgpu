#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

   char *programSource = "__kernel                                            \n"
         "void Mandelbrot(__global float *A,                        \n"
         "            __global float *B,                        \n"
         "            __global int *C)                        \n"
         "{                                                   \n"
         "   int idx = get_global_id(0);                      \n"
         "   int c; int Z;                                  \n"
         "   c = (A[idx]*A[idx]) + (B[idx]*B[idx]);           \n"
         "   Z = 0;                                           \n"
         "   for (int j = 0; j <100; j++){                     \n"
         "   Z = (Z*Z) + c;                                   \n"
         "   }                                                \n"
         "   if ( Z > 2 ) C[idx] = 1;                         \n"
         "   else C[idx] = 0;                                 \n"
         "}                                                   \n";
;

void elementsA(float* data, int size)
{

    int half = size/2;
    for (int j = 0; j < size; j ++){
        for (int i = 0; i<half; i++) data[(j * size) + i] = float(((-2.0) *(half-i))/half);
        for (int i = half; i < size; i++) data[(j * size) + i] = float((2.0 *(i-half))/half);
    }
}
void elementsB(float* data, int size)
{

    int half = size/2;
    for (int j = 0; j < size; j ++){
        for (int i = 0; i<half; i++) data[(i * size) + j] = float((2.0 *(half-i))/half);
        for (int i = half; i < size; i++) data[(i * size) + j] = float(((-2.0) *(i-half))/half);
    }
}

int main() {

    float *matrixA = NULL;
	float *matrixB = NULL;
	int *matrixC = NULL;
    int size;
    scanf("%d", &size);
	int totalElements = size*size;

	size_t datasizeTotal = sizeof(float)*totalElements;

	matrixA = (float*)malloc(datasizeTotal);
	matrixB = (float*)malloc(datasizeTotal);
	matrixC = (int*)malloc(datasizeTotal);

	elementsA(matrixA, size);
	elementsA(matrixB, size);

    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;

    clGetPlatformIDs(0, NULL, &numPlatforms);
 

    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
 

    clGetPlatformIDs(numPlatforms, platforms, NULL);

    
    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    devices = (cl_device_id*) malloc(numDevices*sizeof(cl_device_id));
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);


    cl_context context = NULL;
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, NULL);

    
    cl_command_queue cmdQueue;
    cmdQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

    cl_mem bufferA; 
    cl_mem bufferB;
    cl_mem bufferC;  

    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasizeTotal, NULL, NULL);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasizeTotal, NULL, NULL);
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasizeTotal, NULL, NULL);


    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, NULL);
    clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

    cl_kernel kernel = NULL;

    kernel = clCreateKernel(program, "Mandelbrot", NULL);
			
    clEnqueueWriteBuffer(cmdQueue, bufferA, CL_FALSE, 0, datasizeTotal, matrixA, 0, NULL, NULL);
    clEnqueueWriteBuffer(cmdQueue, bufferB, CL_FALSE, 0, datasizeTotal, matrixB, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    size_t globalWorkSize[1];    
    globalWorkSize[0] = totalElements;

    clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(cmdQueue, bufferC, CL_TRUE, 0, datasizeTotal, matrixC, 0, NULL, NULL);


   printf("\n\nMatrix C: \n");
   for(int i = 0; i < totalElements; i++)
   {
      printf("%d ", matrixC[i]);
      if(((i + 1) % size) == 0) printf("\n");
   }
   printf("\n");
   

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseContext(context);

    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(platforms);
    free(devices);	

}

