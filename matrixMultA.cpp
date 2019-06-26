#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
// OpenCL includes

void randomElements(float* data, int size)
{
   for (int i = 0; i < size; ++i)
   data[i] = rand() % 100;
}



int main(){
	float *matrixA = NULL;
	float *matrixB = NULL;
	float *matrixC = NULL;
    
    int size;
    scanf("%d", &size);

	int totalElements = size*size;

	size_t datasizeTotal = sizeof(float)*totalElements;

	matrixA = (float*)malloc(datasizeTotal);
	matrixB = (float*)malloc(datasizeTotal);
	matrixC = (float*)malloc(datasizeTotal);

	randomElements(matrixA, totalElements);
	randomElements(matrixB, totalElements);
    float count;

	for(int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
            count = 0;
			for (int k = 0; k < size; k++){

				count =  count + matrixA[(i*size) + k] * matrixB[(k*size) + j];
			}
			matrixC[(i*size) + j] = count;
            
		}
	}


    printf("\n\nMatrix C: \n");
   for(int i = 0; i < totalElements; i++)
   {
      printf("%f ", matrixC[i]);
      if(((i + 1) % size) == 0) printf("\n");
   }
   printf("\n");

}