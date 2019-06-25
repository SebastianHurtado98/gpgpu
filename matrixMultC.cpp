#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define WA 1024
#define HA 1024
#define WB 1024
#define HB WA
#define WC WB
#define HC HA

void randomInit(float* data, int size)
{
   for (int i = 0; i < size; ++i)
   data[i] = rand() / (float)RAND_MAX;
}
 
int
main(int argc, char** argv)
{
 
   // set seed for rand()
   srand(2006);
 
   // 1. allocate host memory for matrices A and B
   unsigned int size_A = WA * HA;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* h_A = (float*) malloc(mem_size_A);
 
   unsigned int size_B = WB * HB;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* h_B = (float*) malloc(mem_size_B);
 
   // 2. initialize host memory
   randomInit(h_A, size_A);
   randomInit(h_B, size_B);
 
   // 3. print out A and B
   printf("\n\nMatrix A\n");
   for(int i = 0; i < size_A; i++)
   {
      printf("%f ", h_A[i]);
      if(((i + 1) % WA) == 0)
      printf("\n");
   }
 
   printf("\n\nMatrix B\n");
   for(int i = 0; i < size_B; i++)
   {
      printf("%f ", h_B[i]);
      if(((i + 1) % WB) == 0)
      printf("\n");
   }
 
   // 4. allocate host memory for the result C
   unsigned int size_C = WC * HC;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* h_C = (float*) malloc(mem_size_C);
 
   // 5. Initialize OpenCL
   // OpenCL specific variables
   cl_context clGPUContext;
   cl_command_queue clCommandQue;
   cl_program clProgram;
   cl_kernel clKernel;
  
   size_t dataBytes;
   size_t kernelLength;
   cl_int errcode;
 
   // OpenCL device memory for matrices
   cl_mem d_A;
   cl_mem d_B;
   cl_mem d_C;
 
   /*****************************************/
   /* Initialize OpenCL */
   /*****************************************/
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
          mem_size_A, NULL, &errcode);
   d_A = clCreateBuffer(clGPUContext, 
          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
          mem_size_A, h_A, &errcode);
   d_B = clCreateBuffer(clGPUContext, 
          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
          mem_size_B, h_B, &errcode);
 
 
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
 
   int wA = WA;
   int wC = WC;
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
              d_C, CL_TRUE, 0, mem_size_C, 
              h_C, 0, NULL, NULL);

 
   // 9. print out the results
   printf("\n\nMatrix C (Results)\n");
   for(int i = 0; i < size_C; i++)
   {
      printf("%f ", h_C[i]);
      if(((i + 1) % WC) == 0)
      printf("\n");
   }
   printf("\n");
 
   // 10. clean up memory
   free(h_A);
   free(h_B);
   free(h_C);
 
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
