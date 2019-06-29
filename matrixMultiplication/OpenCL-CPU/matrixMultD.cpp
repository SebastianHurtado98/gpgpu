#include <bits/stdc++.h> 

#define MAX_THREAD 4 

float *matrixA = NULL;
float *matrixB = NULL;
float *matrixC = NULL;

int fixer = 0; 
int size;

void randomElements(float* data, int size)
{
   for (int i = 0; i < size; ++i)
    data[i] = rand() % 100;
}

  
void* multi(void* arg) 
{ 
    int core = fixer++; 
  
    for (int i = core * size / 4; i < (core + 1) * size / 4; i++)  
        for (int j = 0; j < size; j++)  
            for (int k = 0; k < size; k++)  
                matrixC[i*size + j] += matrixA[i*size + k] * matrixB[k*size + j]; 
} 

int main() 
{ 

    scanf("%d", &size);

	int totalElements = size*size;

	size_t datasizeTotal = sizeof(float)*totalElements;

	matrixA = (float*)malloc(datasizeTotal);
	matrixB = (float*)malloc(datasizeTotal);
	matrixC = (float*)malloc(datasizeTotal);

	randomElements(matrixA, totalElements);
	randomElements(matrixB, totalElements);
  
    pthread_t threads[MAX_THREAD]; 
  
    for (int i = 0; i < MAX_THREAD; i++) { 
        int* p; 
        pthread_create(&threads[i], NULL, multi, (void*)(p)); 
    } 
  

    for (int i = 0; i < MAX_THREAD; i++)  
        pthread_join(threads[i], NULL);     
  
    printf("\n\nMatrix C: \n");
   for(int i = 0; i < totalElements; i++)
   {
      printf("%f ", matrixC[i]);
      if(((i + 1) % size) == 0) printf("\n");
   }
   printf("\n");
    return 0; 
} 