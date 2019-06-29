#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
 
void randomElements(float* data, int size)
{
   for (int i = 0; i < size; ++i)
   data[i] = rand() % 100;
}

int main()
{
   int size;
   scanf("%d", &size);
   int totalElements = size*size;
    const int IMAGE_WIDTH = 512;
    const int IMAGE_HEIGHT = 512;
    float zoom = 0.004;
    float offsetX = -0.7;
    float offsetY = 0.0;
    int MAX = 100;
 
   unsigned int sizeC = size * size;
   unsigned int memSizeC = sizeof(float) * sizeC;
   float* C = (float*) malloc(memSizeC);
 
   cl_context clGPUContext;
   cl_command_queue clCommandQue;
   cl_program clProgram;
   cl_kernel clKernel;
  
   size_t dataBytes;
   size_t kernelLength;
   cl_int errcode;
 
   cl_mem bufferA;
   cl_mem bufferB;
   cl_mem bufferC;
 
   clGPUContext = clCreateContextFromType(0, 
                   CL_DEVICE_TYPE_GPU, 
                   NULL, NULL, &errcode);
 
   errcode = clGetContextInfo(clGPUContext, 
              CL_CONTEXT_DEVICES, 0, NULL, 
              &dataBytes);
   cl_device_id *clDevices = (cl_device_id *)
              malloc(dataBytes);
   errcode |= clGetContextInfo(clGPUContext, 
              CL_CONTEXT_DEVICES, dataBytes, 
              clDevices, NULL);

 
   clCommandQue = clCreateCommandQueue(clGPUContext, 
                  clDevices[0], 0, &errcode);

   bufferC = clCreateBuffer(clGPUContext, 
          CL_MEM_READ_WRITE, 
          memSizeC, NULL, &errcode);

   char *clMatrixMul = "__kernel void matrixMul(__global float* C, int zoom, float offsetX, float offsetY, int max, int size){"
   "int tx = get_global_id(0);"
   "int ty = get_global_id(1);"
   "int value = 0; "
    "for (int i = 0; i<max; i++){"
    "    value = value +1 ;}"
    "if (value > 10) value = 1;"
   "C[ty * size + tx] = value;}" ;

   clProgram = clCreateProgramWithSource(clGPUContext, 
                1, (const char **)&clMatrixMul, 
                &kernelLength, &errcode);

   errcode = clBuildProgram(clProgram, 0, 
              NULL, NULL, NULL, NULL);

 
   clKernel = clCreateKernel(clProgram, 
               "matrixMul", &errcode);

 
 
   size_t localWorkSize[2], globalWorkSize[2];
   
   errcode = clSetKernelArg(clKernel, 0, 
              sizeof(cl_mem), (void *)&bufferC);
   errcode = clSetKernelArg(clKernel, 1, 
              sizeof(int), (void *)&zoom);
   errcode = clSetKernelArg(clKernel, 2, 
              sizeof(float), (void *)&offsetX);
   errcode = clSetKernelArg(clKernel, 3, 
              sizeof(float), (void *)&offsetY);
   errcode = clSetKernelArg(clKernel, 4, 
              sizeof(int), (void *)&MAX);
   errcode = clSetKernelArg(clKernel, 5, 
              sizeof(int), (void *)&size);
 
   localWorkSize[0] = 16;
   localWorkSize[1] = 16;
   globalWorkSize[0] = size;
   globalWorkSize[1] = size;
 
   errcode = clEnqueueNDRangeKernel(clCommandQue, 
              clKernel, 2, NULL, globalWorkSize, 
              localWorkSize, 0, NULL, NULL);
 
   errcode = clEnqueueReadBuffer(clCommandQue, 
              bufferC, CL_TRUE, 0, memSizeC, 
              C, 0, NULL, NULL);


   printf("\n\nMatrix C (Results)\n");
   for(int i = 0; i < sizeC; i++)
   {
      printf("%f ", C[i]);
      if(((i + 1) % size) == 0) printf("\n");
   }
   printf("\n");

   

   clReleaseMemObject(bufferC);

   free(clDevices);
   clReleaseContext(clGPUContext);
   clReleaseKernel(clKernel);
   clReleaseProgram(clProgram);
   clReleaseCommandQueue(clCommandQue);

   free(C);
   
 
}
