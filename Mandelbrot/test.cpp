#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
 #include <SFML/Graphics.hpp>
#include <cmath>
#include <random>

const int IMAGE_WIDTH = 512;
const int IMAGE_HEIGHT = 512;
float zoom = 0.004;
float offsetX = -0.7;
float offsetY = 0.0;
int MAX = 100;



/* 
void arrayMandelbrot(int *value)
{
   int size = 512;
   int totalElements = size*size;
 
   unsigned int sizeC = size * size;
   unsigned int memSizeC = sizeof(float) * sizeC;
   float* C = (float*) malloc(memSizeC);
 
   cl_context clGPUContext;
   cl_command_queue clCommandQue;
   cl_program clProgram;
   cl_kernel clKernel;
  
   size_t dataBytes;
   size_t kernelLength;
   cl_int errcode;
 
   cl_mem bufferA;
   cl_mem bufferB;
   cl_mem bufferC;
 
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

   bufferC = clCreateBuffer(clGPUContext, 
          CL_MEM_READ_WRITE, 
          memSizeC, NULL, &errcode);

   char *clMandelbrot = "__kernel void Mandelbrot(__global float* C, float zoom, float offsetX, float offsetY, int max, int size){"
   "int tx = get_global_id(0);"
   "int ty = get_global_id(1);"
    "float startReal = (ty - size / 2.0) * zoom + offsetX;"
    "float startImag = (tx - size / 2.0) * zoom + offsetY;"
    "float zReal = startReal;"
    "float zImag = startImag;"
    "float nextRe;"
    "int counter = 0;"
    "while (  ( (zReal*zReal) + (zImag*zImag) <= 4.0 ) && counter <= 100) {"
    "nextRe = (zReal*zReal) - (zImag * zImag) + startReal;"
    "zImag = 2.0 * zReal * zImag + startImag;"
    "zReal = nextRe;"
    "counter += 1;}"
    "if (counter < max) {"
    "    C[ty * size + tx] = 0;"
    "} else {"
    "    C[ty * size + tx] = 255;}}";

   clProgram = clCreateProgramWithSource(clGPUContext, 
                1, (const char **)&clMandelbrot, 
                &kernelLength, &errcode);

   errcode = clBuildProgram(clProgram, 0, 
              NULL, NULL, NULL, NULL);

 
   clKernel = clCreateKernel(clProgram, 
               "Mandelbrot", &errcode);

 
 
   size_t localWorkSize[2], globalWorkSize[2];
   
   errcode = clSetKernelArg(clKernel, 0, 
              sizeof(cl_mem), (void *)&bufferC);
   errcode = clSetKernelArg(clKernel, 1, 
              sizeof(float), (void *)&zoom);
   errcode = clSetKernelArg(clKernel, 2, 
              sizeof(float), (void *)&offsetX);
   errcode = clSetKernelArg(clKernel, 3, 
              sizeof(float), (void *)&offsetY);
   errcode = clSetKernelArg(clKernel, 4, 
              sizeof(int), (void *)&MAX);
   errcode = clSetKernelArg(clKernel, 5, 
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

   

   clReleaseMemObject(bufferC);

   free(clDevices);
   clReleaseContext(clGPUContext);
   clReleaseKernel(clKernel);
   clReleaseProgram(clProgram);
   clReleaseCommandQueue(clCommandQue);
   
 
}

*/



int main() {
      int size = 512;
   int totalElements = size*size;
 
   unsigned int sizeC = size * size;
   unsigned int memSizeC = sizeof(float) * sizeC;
   float* C = (float*) malloc(memSizeC);
 
   cl_context clGPUContext;
   cl_command_queue clCommandQue;
   cl_program clProgram;
   cl_kernel clKernel;
  
   size_t dataBytes;
   size_t kernelLength;
   cl_int errcode;
 
   cl_mem bufferA;
   cl_mem bufferB;
   cl_mem bufferC;
 
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

   bufferC = clCreateBuffer(clGPUContext, 
          CL_MEM_READ_WRITE, 
          memSizeC, NULL, &errcode);

   char *clMandelbrot = "__kernel void Mandelbrot(__global float* C, float zoom, float offsetX, float offsetY, int max, int size){"
   "int tx = get_global_id(0);"
   "int ty = get_global_id(1);"
    "float startReal = (ty - size / 2.0) * zoom + offsetX;"
    "float startImag = (tx - size / 2.0) * zoom + offsetY;"
    "float zReal = startReal;"
    "float zImag = startImag;"
    "float nextRe;"
    "int counter = 0;"
    "while (  ( (zReal*zReal) + (zImag*zImag) <= 4.0 ) && counter <= 100) {"
    "nextRe = (zReal*zReal) - (zImag * zImag) + startReal;"
    "zImag = 2.0 * zReal * zImag + startImag;"
    "zReal = nextRe;"
    "counter += 1;}"
    "if (counter < max) {"
    "    C[ty * size + tx] = 0;"
    "} else {"
    "    C[ty * size + tx] = 255;}}";

   clProgram = clCreateProgramWithSource(clGPUContext, 
                1, (const char **)&clMandelbrot, 
                &kernelLength, &errcode);

   errcode = clBuildProgram(clProgram, 0, 
              NULL, NULL, NULL, NULL);

 
   clKernel = clCreateKernel(clProgram, 
               "Mandelbrot", &errcode);

 
 
   size_t localWorkSize[2], globalWorkSize[2];
   
   errcode = clSetKernelArg(clKernel, 0, 
              sizeof(cl_mem), (void *)&bufferC);
   errcode = clSetKernelArg(clKernel, 1, 
              sizeof(float), (void *)&zoom);
   errcode = clSetKernelArg(clKernel, 2, 
              sizeof(float), (void *)&offsetX);
   errcode = clSetKernelArg(clKernel, 3, 
              sizeof(float), (void *)&offsetY);
   errcode = clSetKernelArg(clKernel, 4, 
              sizeof(int), (void *)&MAX);
   errcode = clSetKernelArg(clKernel, 5, 
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

   

   clReleaseMemObject(bufferC);
   free(C);
   free(clDevices);
   clReleaseContext(clGPUContext);
   clReleaseKernel(clKernel);
   clReleaseProgram(clProgram);
   clReleaseCommandQueue(clCommandQue);

   

    int elements = IMAGE_HEIGHT * IMAGE_WIDTH;
    int memElements = sizeof(int) * elements;
    int* value = (int*) malloc(memElements);

    sf::RenderWindow window(sf::VideoMode(IMAGE_WIDTH, IMAGE_HEIGHT), "mandel");
    window.setFramerateLimit(30);

    sf::Image image;
    image.create(IMAGE_WIDTH, IMAGE_HEIGHT, sf::Color(0, 0, 0));
    sf::Texture texture;
    sf::Sprite sprite;

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


            //arrayMandelbrot(value);

            for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; i++) {
                int x = int(i/IMAGE_WIDTH);
                int y = int(i%IMAGE_WIDTH);
                image.setPixel(x, y, sf::Color(0, 0, 0));
            }


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