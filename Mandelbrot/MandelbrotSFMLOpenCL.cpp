#include <SFML/Graphics.hpp>
#include <cmath>
#include <random>
//#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
 
const int IMAGE_WIDTH = 512;
const int IMAGE_HEIGHT = 512;
float zoom = 0.004;
float offsetX = -0.7;
float offsetY = 0.0;
const int MAX = 100;

   char *clMandelbrot = "__kernel void Mandelbrot(__global int* values,"
          "int zoom, "
          "int offsetX, "
          "int offsetY, "
          "int maximum, int size){"
        "float startReal = (tx - size / 2.0) * zoom + offsetX;"
        "float startImag = (ty - size / 2.0) * zoom + offsetY;"
        "float zReal = startReal;"
        "float zImag = startImag;"
        "float nextRe;"
        "for (int i = 0; i<maximum; i++){"
        "    nextRe = (zReal*zReal) - (zImag*zImag) + startReal;"
        "    zImag = 2.0 * zReal * zImag + startImag;"
        "    zReal = nextRe;}"
        "if ( ((zReal*zReal) + (zImag*zImag)) <= 4.0)"
        "{values[ty * size + tx] = 0;}"
        "else" 
        "{values[ty * size + tx] = 255;}";
        


int mandelbrot(float, float, int);

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
                int tx = int(i/IMAGE_WIDTH);
                int ty = int(i%IMAGE_WIDTH);
                                float startReal = (tx - 512 / 2.0) * zoom + offsetX;
                float startImag = (ty - 512 / 2.0) * zoom + offsetY;

                    float zReal = startReal;
    float zImag = startImag;
    float nextRe;

    for (int i = 0; i<MAX; i++){
        nextRe = (zReal*zReal) - (zImag*zImag) + startReal;
        zImag = 2.0 * zReal * zImag + startImag;
        zReal = nextRe;
    }
    if ( ((zReal*zReal) + (zImag*zImag)) <= 4.0) {value[i] = 0;}
    else {value[i] = 255;}
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


/*
int mandelbrot(float startReal, float startImag, int maximum) {
    int counter = 0;
    float zReal = startReal;
    float zImag = startImag;
    float nextRe;

    for (int i = 0; i<maximum; i++){
        nextRe = pow(zReal, 2.0) - pow(zImag, 2.0) + startReal;
        zImag = 2.0 * zReal * zImag + startImag;
        zReal = nextRe;
    }
    if (pow(zReal, 2.0) + pow(zImag, 2.0) <= 4.0) return 0;
    return 255;
//WHILE no va si ya tienes el for
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
*/

/*

int main()
{

   int mem = sizeof(int) * 512*512;
   int* value = (int*) malloc(mem);
 
   cl_context clGPUContext;
   cl_command_queue clCommandQue;
   cl_program clProgram;
   cl_kernel clKernel;
  
   size_t dataBytes;
   size_t kernelLength;
   cl_int errcode;
 
   cl_mem bufferValues;
 
   clGPUContext = clCreateContextFromType(0, 
                   CL_DEVICE_TYPE_GPU, 
                   NULL, NULL, &errcode);
 
   errcode = clGetContextInfo(clGPUContext, 
              CL_CONTEXT_DEVICES, 0, NULL, 
              &dataBytes);
   cl_device_id *clDevices = (cl_device_id *)
              malloc(dataBytes);
   errcode |= clGetContextInfo(clGPUContext, 
              CL_CONTEXT_DEVICES, dataBytes, 
              clDevices, NULL);

 
   clCommandQue = clCreateCommandQueue(clGPUContext, 
                  clDevices[0], 0, &errcode);

   bufferValues = clCreateBuffer(clGPUContext, 
          CL_MEM_READ_WRITE, 
          mem, NULL, &errcode);



    //MANDELBROT END

   clProgram = clCreateProgramWithSource(clGPUContext, 
                1, (const char **)&clMatrixMul, 
                &kernelLength, &errcode);

   errcode = clBuildProgram(clProgram, 0, 
              NULL, NULL, NULL, NULL);

 
   clKernel = clCreateKernel(clProgram, 
               "matrixMul", &errcode);

 
 
   size_t localWorkSize[2], globalWorkSize[2];
   
   errcode = clSetKernelArg(clKernel, 0, 
              sizeof(cl_mem), (void *)&bufferC);
   errcode |= clSetKernelArg(clKernel, 1, 
              sizeof(cl_mem), (void *)&bufferA);
   errcode |= clSetKernelArg(clKernel, 2, 
              sizeof(cl_mem), (void *)&bufferB);
   errcode |= clSetKernelArg(clKernel, 3, 
              sizeof(int), (void *)&size);
   errcode |= clSetKernelArg(clKernel, 4, 
              sizeof(int), (void *)&size);
 
   localWorkSize[0] = 16;
   localWorkSize[1] = 16;
   globalWorkSize[0] = size;
   globalWorkSize[1] = size;
 
   errcode = clEnqueueNDRangeKernel(clCommandQue, 
              clKernel, 2, NULL, globalWorkSize, 
              localWorkSize, 0, NULL, NULL);
 
   errcode = clEnqueueReadBuffer(clCommandQue, 
              bufferC, CL_TRUE, 0, memSizeC, 
              C, 0, NULL, NULL);


   printf("\n\nMatrix C (Results)\n");
   for(int i = 0; i < sizeC; i++)
   {
      printf("%f ", C[i]);
      if(((i + 1) % size) == 0) printf("\n");
   }
   printf("\n");

   
   clReleaseMemObject(bufferA);
   clReleaseMemObject(bufferC);
   clReleaseMemObject(bufferB);

   free(clDevices);
   clReleaseContext(clGPUContext);
   clReleaseKernel(clKernel);
   clReleaseProgram(clProgram);
   clReleaseCommandQueue(clCommandQue);

   free(A);
   free(B);
   free(C);
   
 
}


 */