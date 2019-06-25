#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
// OpenCL includes


int main(){
	int *matrixA = NULL;
	int *matrixB = NULL;
	int *matrixC = NULL;
    
    int size = 1024;
    //scanf("%d", &size);

	const int totalElements = size*size;

	size_t datasizeTotal = sizeof(int)*totalElements;

	matrixA = (int*)malloc(datasizeTotal);
	matrixB = (int*)malloc(datasizeTotal);
	matrixC = (int*)malloc(datasizeTotal);

	for(int i = 0; i < totalElements; i++) {
	matrixA[i] = 2;
	matrixB[i] = 2;
	}
    int count;

	for(int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
            count = 0;
			for (int k = 0; k < size; k++){

				count =  count + matrixA[(i*size) + k] * matrixB[(k*size) + j];
			}
			matrixC[(i*size) + j] = count;
            
		}
	}


    for (int i = 0 ; i<size; i++) for (int j=0; j<size; j++) printf("%d ", matrixC[(i*size) + j]);
    printf("\n");

}