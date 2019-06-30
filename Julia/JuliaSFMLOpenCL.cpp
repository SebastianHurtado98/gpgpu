#include <iostream>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sstream>
#include <sys/stat.h>
#include <CL/cl.h>
#include <vector>
#include "SFML/Graphics.hpp"

const int width = 1280;
const int height = 720;
const int dataSize = 1280*720;
std::vector<int> framesPerSecond;

class FPS
{ 
public:
	FPS() : mFrame(0), mFps(0) {}
	const unsigned int getFPS() const { return mFps; }

private:
	unsigned int mFrame;
	unsigned int mFps;
	sf::Clock mClock;

public:
	void update()
	{
		if(mClock.getElapsedTime().asSeconds() >= 1.f)
		{
			mFps = mFrame;
			mFrame = 0;
			mClock.restart();
		}
 
		++mFrame;
	}
};


void JuliaSet(sf::VertexArray& vertexarray, float* results)
{
    for(int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int iterations = results[i * width + j];

            vertexarray[i*width + j].position = sf::Vector2f(j, i);
            sf::Color color(iterations%256, iterations%256, iterations%256);
            vertexarray[i*width + j].color = color;
        }
    }
}


const char *KernelSource = "\n" \
"__kernel void Calculate_Julia_Set(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int dataSize, int pixel_shift_x, int pixel_shift_y, int precision, float zoom, int x, int y)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   int j = i % 1280; int k = i / 1280;                                 \n" \
"   float c_real = ((float)x) / zoom  - pixel_shift_x / 1280.0f;                    \n" \
"   float c_imag = ((float)y) / zoom  - pixel_shift_y / 720.0f;                    \n" \
"   float z_real = ((float)j) / zoom  - pixel_shift_x / 1280.0f;                                               \n" \
"   float z_imag = ((float)k) / zoom  - pixel_shift_y / 720.0f; int iterations = 0;                             \n" \
"   if(i < dataSize)                                                       \n" \
"   for (int l = 0; l < precision; l++)                                 \n" \
"   {                                                                   \n" \
"       float z1_real = z_real * z_real - z_imag * z_imag; float z1_imag = 2 * z_real * z_imag; \n" \
"       z_real = z1_real + c_real; z_imag = z1_imag + c_imag; iterations++;                                 \n" \
"       if (z_real * z_real + z_imag * z_imag > 4) { break; }                                \n" \
"   }                                                                   \n" \
"   output[i] = iterations;                                             \n" \
"}                                                                      \n" \
"\n";


int main()
{
    int err;                          
    int memElements = sizeof(int) * dataSize;
    float* data = (float*) malloc(memElements);
    float* results = (float*) malloc(memElements);
    
    size_t global;                     
    size_t local;                      
    
    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;
    
    cl_mem input;              
    cl_mem output;
    
    
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    
    context = clCreateContext(0, 1, &device_id, NULL, NULL, NULL);

    commands = clCreateCommandQueue(context, device_id, 0, NULL);

    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, NULL);

    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    
    kernel = clCreateKernel(program, "Calculate_Julia_Set", NULL);

    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * dataSize, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * dataSize, NULL, NULL);
    
    clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * dataSize, data, 0, NULL, NULL);
    
    sf::RenderWindow window(sf::VideoMode(width, height), "Mandelbrot - Julia");
    window.setFramerateLimit(60);
    sf::VertexArray pointmap(sf::Points, width * height);
    

    float zoom = 275.0f;
    int precision = 300;
    int x_shift = width * 2.5;
    int y_shift = height * 1.2;
    
    clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    
    for (int i = 0; i < width*height; i++)
    {
        pointmap[i].color = sf::Color::Black;
    }
    

    clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * dataSize, data, 0, NULL, NULL);
    
    int mouse_x = sf::Mouse::getPosition().x;
    int mouse_y = sf::Mouse::getPosition().y;
    
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &dataSize);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &x_shift);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &y_shift);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &precision);
    err |= clSetKernelArg(kernel, 6, sizeof(float), &zoom);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &mouse_x);
    err |= clSetKernelArg(kernel, 8, sizeof(float), &mouse_y);
    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    
    global = dataSize;
    
    clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
   
    clFinish(commands);
    
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * dataSize, results, 0, NULL, NULL );
    
    JuliaSet(pointmap, results);

    FPS fps;
    
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        
        mouse_x = sf::Mouse::getPosition().x;
        mouse_y = sf::Mouse::getPosition().y;
        
        //pantalla negra
        for (int i = 0; i < width*height; i++)
        {
            pointmap[i].color = sf::Color::Black;
        }
        

        clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * dataSize, data, 0, NULL, NULL);

        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
        err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &dataSize);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &x_shift);
        err |= clSetKernelArg(kernel, 4, sizeof(int), &y_shift);
        err |= clSetKernelArg(kernel, 5, sizeof(int), &precision);
        err |= clSetKernelArg(kernel, 6, sizeof(float), &zoom);
        err |= clSetKernelArg(kernel, 7, sizeof(int), &mouse_x);
        err |= clSetKernelArg(kernel, 8, sizeof(float), &mouse_y);
        
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set kernel arguments! %d\n", err);
            exit(1);
        }
        

        global = dataSize;
        clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);

        clFinish(commands);
        
        clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * dataSize, results, 0, NULL, NULL );
        
        JuliaSet(pointmap, results);
        
        window.clear();
        window.draw(pointmap);
        window.display();

        fps.update();
        std::ostringstream ss;
        ss << fps.getFPS();
        framesPerSecond.push_back(fps.getFPS());

        window.setTitle(ss.str());

    }

        for(auto f : framesPerSecond){
        printf("%d \n", f);
    }
    
    return 0;
}