#include <stdio.h>
#include <cuda_runtime.h>
#include "kernel.h"

__global__ void calculateJuliaSet(float *data, int dataSize,
    int mouse_x, int mouse_y, float zoom, int x_shift, int y_shift,
    int precision) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = i % 1280;
    int k = i / 1280;  

    float c_real = ((float)mouse_x) / zoom - x_shift / 1280.0f;    
    float c_imag = ((float)mouse_y) / zoom - y_shift / 720.0f;   
    float z_real = ((float)j) / zoom - x_shift / 1280.0f;     
    float z_imag = ((float)k) / zoom - y_shift / 720.0f; 
    int iterations = 0;   

    if (i < dataSize)
    for (int l = 0; l < precision; l++) {          
        float z1_real = z_real * z_real - z_imag * z_imag;
        float z1_imag = 2 * z_real * z_imag;
        z_real = z1_real + c_real;
        z_imag = z1_imag + c_imag;
        iterations++;  

        if ((z_real * z_real + z_imag * z_imag) > 4) { 
            break; 
        }
    }

    data[i] = iterations;
}

void setJuliaSet(float *data, int dataSize, int mouse_x, int mouse_y, float zoom, int x_shift, int y_shift, int precision) {
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(1280 / blockDim.x, 720 / blockDim.y, 1);

    cudaMallocManaged(&data, dataSize * sizeof(float));

    calculateJuliaSet<<<blockDim, gridDim, 0>>>(data, dataSize, mouse_x, mouse_y, zoom, x_shift, y_shift, precision);
}