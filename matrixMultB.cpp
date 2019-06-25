
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <CL/cl.h>

const char* programSource =
"__kernel                                            \n"
"void vecadd(__global int *A,                        \n"
"            __global int *B,                        \n"
"            __global int *C)                        \n"
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

int vectMultiplication(int *matrixA, int *matrixB) {


    int *A = NULL; 
    int *B = NULL; 
    int *C = NULL;
    

    const int elements = 32;   
    
    size_t datasize = sizeof(int)*elements;

    A = (int*)malloc(datasize);
    B = (int*)malloc(datasize);
    C = (int*)malloc(datasize);

    for(int i = 0; i < elements; i++) {
        A[i] = matrixA[i]; 
        B[i] = matrixB[i]; 
    }

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

    kernel = clCreateKernel(program, "vecadd", &status);

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
    globalWorkSize[0] = elements;

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


    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseContext(context);

	int count = 0;
	for(int i = 0; i < 32 ; i++){
		count = count + C[i];
	}

    free(A);
    free(B);
    free(C);
    free(platforms);
    free(devices);

	return count;

}



int main(){
	int *matrixA = NULL;
	int *matrixB = NULL;
	int *matrixC = NULL;


	const int totalElements = 1024;

	size_t datasizeTotal = sizeof(int)*totalElements;

	matrixA = (int*)malloc(datasizeTotal);
	matrixB = (int*)malloc(datasizeTotal);
	matrixC = (int*)malloc(datasizeTotal);

	for(int i = 0; i < totalElements; i++) {
	    matrixA[i] = i;
	    matrixB[i] = i;
	}


	size_t datasizeVect = sizeof(int)*32;
	int *vectA = NULL;
	int *vectB = NULL;
	vectA = (int*)malloc(datasizeVect);
	vectB = (int*)malloc(datasizeVect);

	for(int i = 0; i < 32; i++){
		for (int j = 0; j < 32; j++){
			for (int k = 0; k < 32; k++){

				vectA[k] = matrixA[(i*32) + k];
				vectB[k] = matrixB[(k*32) + j];
			}
			printf("%d \n", (i*32)+j);
			matrixC[(i*32) + j] = vectMultiplication(vectA, vectB);
		}
	}

	for(int i =0; i<32; i++){
		for(int j = 0; j<32; j++){
			printf("%d", matrixC[(i*32) + j]);
		}
	}

}