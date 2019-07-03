#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "kernel.cu"
#include "array.h"
#include <math.h>

using namespace std;

int main()
{
    int N = 4096;
    int SIZE = N*N;

    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);
    vector<float> h_C(SIZE);

    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = sin(i);
            h_B[i*N+j] = cos(j);
        }
    }

    array<float> d_A(SIZE);
    array<float> d_B(SIZE);
    array<float> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    cudaDeviceSynchronize();

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();

    return 0;
}