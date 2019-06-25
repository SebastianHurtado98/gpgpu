#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
 
void randomInit(float* data, int size)
{
   for (int i = 0; i < size; ++i)
   data[i] = 1;
}

int main()
{
   int size = 1024;
 
   // 1. allocate host memory for matrices A and B
   unsigned int sizeA = size * size;
   unsigned int memSizeA = sizeof(float) * sizeA;
   float* A = (float*) malloc(memSizeA);
 
   unsigned int sizeB = size * size;
   unsigned int memSizeB = sizeof(float) * sizeB;
   float* B = (float*) malloc(memSizeB);

   randomInit(A, sizeA);
   randomInit(B, sizeB);

 
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
 
   cl_mem d_A;
   cl_mem d_B;
   cl_mem d_C;
 
   clGPUContext = clCreateContextFromType(0, 
                   CL_DEVICE_TYPE_GPU, 
                   NULL, NULL, &errcode);
 
   // get the list of GPU devices associated 
   // with context
   errcode = clGetContextInfo(clGPUContext, 
              CL_CONTEXT_DEVICES, 0, NULL, 
              &dataBytes);
   cl_device_id *clDevices = (cl_device_id *)
              malloc(dataBytes);
   errcode |= clGetContextInfo(clGPUContext, 
              CL_CONTEXT_DEVICES, dataBytes, 
              clDevices, NULL);

 
   //Create a command-queue
   clCommandQue = clCreateCommandQueue(clGPUContext, 
                  clDevices[0], 0, &errcode);

   // Setup device memory
   d_C = clCreateBuffer(clGPUContext, 
          CL_MEM_READ_WRITE, 
          memSizeC, NULL, &errcode);
   d_A = clCreateBuffer(clGPUContext, 
          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
          memSizeA, A, &errcode);
   d_B = clCreateBuffer(clGPUContext, 
          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
          memSizeB, B, &errcode);
 
 
   // 6. Load and build OpenCL kernel
   char *clMatrixMul = "__kernel void matrixMul(__global float* C,"
          "__global float* A, "
          "__global float* B, "
          "int wA, int wB){"
   "int tx = get_global_id(0);"
   "int ty = get_global_id(1);"
   "float value = 0;"
   "for (int k = 0; k < wA; ++k){"
      "float elementA = A[ty * wA + k];"
      "float elementB = B[k * wB + tx];"
      "value += elementA * elementB;}"
   "C[ty * wA + tx] = value;}" ;

   clProgram = clCreateProgramWithSource(clGPUContext, 
                1, (const char **)&clMatrixMul, 
                &kernelLength, &errcode);

 
   errcode = clBuildProgram(clProgram, 0, 
              NULL, NULL, NULL, NULL);

 
   clKernel = clCreateKernel(clProgram, 
               "matrixMul", &errcode);

 
 
   // 7. Launch OpenCL kernel
   size_t localWorkSize[2], globalWorkSize[2];
 
   int wA = size;
   int wC = size;
   errcode = clSetKernelArg(clKernel, 0, 
              sizeof(cl_mem), (void *)&d_C);
   errcode |= clSetKernelArg(clKernel, 1, 
              sizeof(cl_mem), (void *)&d_A);
   errcode |= clSetKernelArg(clKernel, 2, 
              sizeof(cl_mem), (void *)&d_B);
   errcode |= clSetKernelArg(clKernel, 3, 
              sizeof(int), (void *)&wA);
   errcode |= clSetKernelArg(clKernel, 4, 
              sizeof(int), (void *)&wC);
 
   localWorkSize[0] = 16;
   localWorkSize[1] = 16;
   globalWorkSize[0] = 1024;
   globalWorkSize[1] = 1024;
 
   errcode = clEnqueueNDRangeKernel(clCommandQue, 
              clKernel, 2, NULL, globalWorkSize, 
              localWorkSize, 0, NULL, NULL);
 
   // 8. Retrieve result from device
   errcode = clEnqueueReadBuffer(clCommandQue, 
              d_C, CL_TRUE, 0, memSizeC, 
              C, 0, NULL, NULL);

 
   // 9. print out the results
   printf("\n\nMatrix C (Results)\n");
   for(int i = 0; i < sizeC; i++)
   {
      printf("%f ", C[i]);
      if(((i + 1) % size) == 0)
      printf("\n");
   }
   printf("\n");
 
   // 10. clean up memory
   free(A);
   free(B);
   free(C);
 
   clReleaseMemObject(d_A);
   clReleaseMemObject(d_C);
   clReleaseMemObject(d_B);
 
   free(clDevices);
   free(clMatrixMul);
   clReleaseContext(clGPUContext);
   clReleaseKernel(clKernel);
   clReleaseProgram(clProgram);
   clReleaseCommandQueue(clCommandQue);

}
