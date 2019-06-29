#include <SFML/Graphics.hpp>
#include <cmath>
#include <random>

const int width = 1280;
const int height = 720;
const int dataSize = 1280*720;
float zoom = 275.0f;
int precision = 300;
int x_shift = width * 2.5;
int y_shift = height * 1.2;


void JuliaSet(sf::VertexArray& vertexarray, float* data, int x, int y)
{

    for (int i = 0; i <dataSize; i++){                                
        int j = i % 1280; int k = i / 1280;                              
        float c_real = ((float)x) / zoom  - x_shift / 1280.0f;                
        float c_imag = ((float)y) / zoom  - y_shift / 720.0f;                    
        float z_real = ((float)j) / zoom  - x_shift / 1280.0f;                                             
        float z_imag = ((float)k) / zoom  - y_shift / 720.0f; 
        int iterations = 0;                           
        if(i < dataSize)                                                      
        for (int l = 0; l < precision; l++)                               
        {                                                                  
            float z1_real = z_real * z_real - z_imag * z_imag; float z1_imag = 2 * z_real * z_imag;
            z_real = z1_real + c_real; z_imag = z1_imag + c_imag; iterations++;                               
            if (z_real * z_real + z_imag * z_imag > 4) { break; }                                
        }                                                                   
        data[i] = iterations;                                     
        }                                                 
    for(int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int iterations = data[i * width + j];

            vertexarray[i*width + j].position = sf::Vector2f(j, i);
            sf::Color color(iterations%256, iterations%256, iterations%256);
            vertexarray[i*width + j].color = color;
        }
    }
}

int main()
{
    sf::RenderWindow window(sf::VideoMode(width, height), "Mandelbrot - Julia");
    window.setFramerateLimit(30);
    sf::VertexArray pointmap(sf::Points, width * height);
                       
    int memElements = sizeof(int) * dataSize;
    float* data = (float*) malloc(memElements);
    

    
    for (int i = 0; i < width*height; i++)
    {
        pointmap[i].color = sf::Color::Black;
    }

    int mouse_x = sf::Mouse::getPosition().x;
    int mouse_y = sf::Mouse::getPosition().y;
    
    
    JuliaSet(pointmap, data, mouse_x, mouse_y);
    
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
        

        JuliaSet(pointmap, data, mouse_x, mouse_y);
        
        
        window.clear();
        window.draw(pointmap);
        window.display();
    }
    
    return 0;
}

