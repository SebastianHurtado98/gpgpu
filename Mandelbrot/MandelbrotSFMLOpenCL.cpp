#include <SFML/Graphics.hpp>
#include <cmath>
#include <random>
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
 
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

const int IMAGE_WIDTH = 512;
const int IMAGE_HEIGHT = 512;
double zoom = 0.004;
double offsetX = -0.7;
double offsetY = 0.0;
const int MAX = 100;

int mandelbrot(double, double, int);
sf::Color getColor(int);

int main() {
    sf::RenderWindow window(sf::VideoMode(IMAGE_WIDTH, IMAGE_HEIGHT), "Mandelbrot");
    window.setFramerateLimit(30);

    sf::Image image;
    image.create(IMAGE_WIDTH, IMAGE_HEIGHT, sf::Color(0, 0, 0));
    sf::Texture texture;
    sf::Sprite sprite;



    int elements = IMAGE_HEIGHT * IMAGE_WIDTH;
    int memElements = sizeof(int) * elements;
    int* value = (int*) malloc(memElements);



    bool stateChanged = true; 

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            switch (event.type) {
                case sf::Event::Closed:
                    window.close();
                    break;
                case sf::Event::KeyPressed:
                    stateChanged = true; 
                    switch (event.key.code) {
                        case sf::Keyboard::Escape:
                            window.close();
                            break;
                        case sf::Keyboard::Q:
                            zoom *= 0.9;
                            break;
                        case sf::Keyboard::E:
                            zoom /= 0.9;
                            break;
                        case sf::Keyboard::W:
                            offsetY -= 40 * zoom;
                            break;
                        case sf::Keyboard::S:
                            offsetY += 40 * zoom;
                            break;
                        case sf::Keyboard::A:
                            offsetX -= 40 * zoom;
                            break;
                        case sf::Keyboard::D:
                            offsetX += 40 * zoom;
                            break;
                        default: break;
                    }
                default:
                    break;
            }
        }


        if (stateChanged) {

            for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; i++){
                int x = int(i/IMAGE_WIDTH);
                int y = int(i%IMAGE_WIDTH);
                double real = (x - IMAGE_WIDTH / 2.0) * zoom + offsetX;
                double imag = (y - IMAGE_HEIGHT / 2.0) * zoom + offsetY;
                value[i] = mandelbrot(real, imag, MAX);
            }

            for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; i++){
                int x = int(i/IMAGE_WIDTH);
                int y = int(i%IMAGE_WIDTH);
                image.setPixel(x, y, sf::Color(value[i], value[i], value[i]));
            }




            /*
            for (int x = 0; x < IMAGE_WIDTH; x++) {
                for (int y = 0; y < IMAGE_HEIGHT; y++) {
                    double real = (x - IMAGE_WIDTH / 2.0) * zoom + offsetX;
                    double imag = (y - IMAGE_HEIGHT / 2.0) * zoom + offsetY;
                    int value = mandelbrot(real, imag, MAX);
                    image.setPixel(x, y, sf::Color(value, value, value));
                }
            }
            */

            texture.loadFromImage(image);
            sprite.setTexture(texture);
        }


        window.clear();
        window.draw(sprite);
        window.display();

        stateChanged = false;
    }

    return 0;
}

int mandelbrot(double startReal, double startImag, int maximum) {
    int counter = 0;
    double zReal = startReal;
    double zImag = startImag;
    double nextRe;

    while (pow(zReal, 2.0) + pow(zImag, 2.0) <= 4.0 && counter <= maximum) {
        nextRe = pow(zReal, 2.0) - pow(zImag, 2.0) + startReal;
        zImag = 2.0 * zReal * zImag + startImag;
        zReal = nextRe;
        if (zReal == startReal && zImag == startImag) { 
            return 0;
        }
        counter += 1;
    }

    if (counter >= maximum) {
        return 0;
    } else {
        return 255; 
    }
}

/*

int main() {

    int *value = NULL;
	int totalElements = 512*512;

	size_t datasizeTotal = sizeof(float)*totalElements;

	value = (float*)malloc(datasizeTotal);

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



 */