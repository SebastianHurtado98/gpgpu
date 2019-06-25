
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <CL/cl.h>

const char* programSource =
"__kernel                                            \n"
"void vectMult(__global float *A,                        \n"
"            __global float *B,                        \n"
"            __global float *C)                        \n"
"{                                                   \n"
"                                                    \n"
"   // Get the work-item’s unique ID                 \n"
"   int idx = get_global_id(0);                      \n"
"                                                    \n"
"   // Add the corresponding locations of            \n"
"   // 'A' and 'B', and store the result in 'C'.     \n"
"   C[idx] = A[idx] * B[idx];                        \n"
"}                                                   \n"
;

int main() {

    float *matrixA = NULL;
	float *matrixB = NULL;
	float *matrixC = NULL;

    int size = 32;   
	int totalElements = size*size;

	size_t datasizeTotal = sizeof(float)*totalElements;

	matrixA = (float*)malloc(datasizeTotal);
	matrixB = (float*)malloc(datasizeTotal);
	matrixC = (float*)malloc(datasizeTotal);

	for(int i = 0; i < totalElements; i++) {
	    matrixA[i] = 10;
	    matrixB[i] = 10;
	}

    float *A = NULL; 
    float *B = NULL; 
    float *C = NULL;
    


    
    size_t datasize = sizeof(float)*size;

    A = (float*)malloc(datasize);
    B = (float*)malloc(datasize);
    C = (float*)malloc(datasize);



    cl_int status;  
     
    
    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;

    status = clGetPlatformIDs(0, NULL, &numPlatforms);
 

    platforms =   
        (cl_platform_id*)malloc(
            numPlatforms*sizeof(cl_platform_id));
 

    status = clGetPlatformIDs(numPlatforms, platforms, 
                NULL);

    
    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;
    status = clGetDeviceIDs(
        platforms[0], 
        CL_DEVICE_TYPE_ALL, 
        0, 
        NULL, 
        &numDevices);


    devices = 
        (cl_device_id*)malloc(
            numDevices*sizeof(cl_device_id));

    status = clGetDeviceIDs(
        platforms[0], 
        CL_DEVICE_TYPE_ALL,        
        numDevices, 
        devices, 
        NULL);


    cl_context context = NULL;

    context = clCreateContext(
        NULL, 
        numDevices, 
        devices, 
        NULL, 
        NULL, 
        &status);

    
    cl_command_queue cmdQueue;

    cmdQueue = clCreateCommandQueue(
        context, 
        devices[0], 
        0, 
        &status);

    cl_mem bufferA; 
    cl_mem bufferB;
    cl_mem bufferC;  


    bufferA = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY,                         
        datasize, 
        NULL, 
        &status);

    bufferB = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY,                         
        datasize, 
        NULL, 
        &status);

    bufferC = clCreateBuffer(
        context, 
        CL_MEM_WRITE_ONLY,                 
        datasize, 
        NULL, 
        &status);


    cl_program program = clCreateProgramWithSource(
        context, 
        1, 
        (const char**)&programSource,                                 
        NULL, 
        &status);

    status = clBuildProgram(
        program, 
        numDevices, 
        devices, 
        NULL, 
        NULL, 
        NULL);

    cl_kernel kernel = NULL;

    kernel = clCreateKernel(program, "vectMult", &status);


    //Repetible:
    for(int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			for (int k = 0; k < size; k++){
				A[k] = matrixA[(i*size) + k];
				B[k] = matrixB[(k*size) + j];
			}

            
			
        status = clEnqueueWriteBuffer(
            cmdQueue, 
            bufferA, 
            CL_FALSE, 
            0, 
            datasize,                         
            A, 
            0, 
            NULL, 
            NULL);
        
        status = clEnqueueWriteBuffer(
            cmdQueue, 
            bufferB, 
            CL_FALSE, 
            0, 
            datasize,                                  
            B, 
            0, 
            NULL, 
            NULL);

        status  = clSetKernelArg(
            kernel, 
            0, 
            sizeof(cl_mem), 
            &bufferA);
        status |= clSetKernelArg(
            kernel, 
            1, 
            sizeof(cl_mem), 
            &bufferB);
        status |= clSetKernelArg(
            kernel, 
            2, 
            sizeof(cl_mem), 
            &bufferC);

        size_t globalWorkSize[1];    
        globalWorkSize[0] = size;

        status = clEnqueueNDRangeKernel(
            cmdQueue, 
            kernel, 
            1, 
            NULL, 
            globalWorkSize, 
            NULL, 
            0, 
            NULL, 
            NULL);


        clEnqueueReadBuffer(
            cmdQueue, 
            bufferC, 
            CL_TRUE, 
            0, 
            datasize, 
            C, 
            0, 
            NULL, 
            NULL);

        float count = 0;
        for(int w = 0; w < size ; w++){
            count = count + C[w];
        }

        matrixC[(i*size) + j] = count;
		}
	}
    //fin de repetible
    //AL FINAL!!

   printf("\n\nMatrix C: \n");
   for(int i = 0; i < totalElements; i++)
   {
      printf("%f ", matrixC[i]);
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

    free(A);
    free(B);
    free(C);
    free(platforms);
    free(devices);	

}

