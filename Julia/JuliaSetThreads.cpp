#include <bits/stdc++.h> 

#define MAX_THREAD 4 

float *matrixA = NULL;
float *matrixB = NULL;
int *matrixC = NULL;
int iterations = 10;

int fixer = 0; 
int size;

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

  
void* Mandelbrot(void* arg) 
{ 
    int core = fixer++; 
    int totalSize = size * size;
    int Z = 0;
    int C;

    for (int i = core * totalSize / 4; i < (core + 1) * totalSize / 4; i++)  {
        Z = 0;
        C = (matrixA[i] * matrixA[i]) + (matrixB[i] * matrixB[i]);
        for (int j = 0; j < iterations; j++){
            Z = (Z*Z) + C;
        }
        if ( Z > 2 ) matrixC[i] = 1;
        else matrixC[i] = 0;
    }
} 

int main() 
{ 

    scanf("%d", &size);

	int totalElements = size*size;

	size_t datasizeTotal = sizeof(float)*totalElements;

	matrixA = (float*)malloc(datasizeTotal);
	matrixB = (float*)malloc(datasizeTotal);
	matrixC = (int*)malloc(datasizeTotal);

	elementsA(matrixA, size);
	elementsB(matrixB, size);
  

    pthread_t threads[MAX_THREAD]; 
  
    for (int i = 0; i < MAX_THREAD; i++) { 
        int* p; 
        pthread_create(&threads[i], NULL, Mandelbrot, (void*)(p)); 
    } 
  

    for (int i = 0; i < MAX_THREAD; i++)  
        pthread_join(threads[i], NULL);     

    printf("\n\nMatrix C: \n");
   for(int i = 0; i < totalElements; i++)
   {
      printf("%d ", matrixC[i]);
      if(((i + 1) % size) == 0) printf("\n");
   }
   printf("\n");
    return 0; 
} 