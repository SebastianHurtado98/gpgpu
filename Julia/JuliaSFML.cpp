#include <SFML/Graphics.hpp>
#include <cmath>
#include <sstream>

const int width = 1280;
const int height = 720;
const int dataSize = 1280*720;
float zoom = 275.0f;
int maximum = 300;
int x_shift = width * 2.5;
int y_shift = height * 1.2;
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


void JuliaSet(sf::VertexArray& vertexarray, float* data, int x, int y)
{

    for (int i = 0; i <dataSize; i++){                                
        int j = i % 1280; int k = i / 1280;                              
        float cReal = ((float)x) / zoom  - x_shift / 1280.0f;                
        float cImag = ((float)y) / zoom  - y_shift / 720.0f;                    
        float zReal = ((float)j) / zoom  - x_shift / 1280.0f;                                             
        float zImag = ((float)k) / zoom  - y_shift / 720.0f; 
        int counter = 0;                           
        if(i < dataSize)                                                      
        for (int j = 0; j < maximum; j++)                               
        {                                                                  
            float nextReal = zReal * zReal - zImag * zImag; 
            float z1Imag = 2 * zReal * zImag;
            zReal = nextReal + cReal; 
            zImag = z1Imag + cImag; 
            counter++;                               
            if (zReal * zReal + zImag * zImag > 4.0) { break; }                                
        }                                                                   
        data[i] = counter;                                     
        }                                                 
    for(int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int counter = data[i * width + j];

            vertexarray[i*width + j].position = sf::Vector2f(j, i);
            sf::Color color(counter%256, counter%256, counter%256);
            vertexarray[i*width + j].color = color;
        }
    }
}

int main()
{
    sf::RenderWindow window(sf::VideoMode(width, height), "Julia");
    window.setFramerateLimit(60);
    sf::VertexArray pixels(sf::Points, width * height);
                       
    int memElements = sizeof(int) * dataSize;
    float* data = (float*) malloc(memElements);
    

    
    for (int i = 0; i < width*height; i++)
    {
        pixels[i].color = sf::Color::Black;
    }

    int mouse_x = sf::Mouse::getPosition().x;
    int mouse_y = sf::Mouse::getPosition().y;
    
    
    JuliaSet(pixels, data, mouse_x, mouse_y);

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
        
        //black (empty) screen
        for (int i = 0; i < width*height; i++)
        {
            pixels[i].color = sf::Color::Black;
        }
        
        JuliaSet(pixels, data, mouse_x, mouse_y);

        window.clear();
        window.draw(pixels);
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

