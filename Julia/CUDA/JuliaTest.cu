#include <SFML/Graphics.hpp>
#include <cmath>
#include <random>
#include <sstream>
#include <cuda_runtime.h>
#include "kernel.h"

int width = 1280;
int height = 720;
int dataSize = 1280*720;

std::vector<int> framesPerSecond;

class FPS {
public:
	FPS() : mFrame(0), mFps(0) {}
	unsigned int getFPS() { return mFps; }
private:
	unsigned int mFrame;
	unsigned int mFps;
	sf::Clock mClock;
public:
	void update() {
		if(mClock.getElapsedTime().asSeconds() >= 1.f) {
			mFps = mFrame;
			mFrame = 0;
			mClock.restart();
		}
		++mFrame;
	}
};

void juliaSet(sf::VertexArray &vertexarray, float *results, int height, int width) {
    for(int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int iterations = results[i * width + j];
            vertexarray[i * width + j].position = sf::Vector2f(j, i);
            sf::Color color(iterations % 256, iterations % 256, iterations % 256);
            vertexarray[i * width + j].color = color;
        }
    }
}

int main() {
    int memElements = sizeof(int) * dataSize;
    float *data = (float*) malloc(memElements);

    sf::RenderWindow window(sf::VideoMode(width, height), "Mandelbrot - Julia");
    window.setFramerateLimit(60);
    sf::VertexArray pointmap(sf::Points, dataSize);

    float zoom = 275.0f;
    int precision = 300;
    int x_shift = width * 2.5;
    int y_shift = height * 1.2;

    for (int i = 0; i < dataSize; i++) {
        pointmap[i].color = sf::Color::Green;
    }

    int mouse_x = sf::Mouse::getPosition().x;
    int mouse_y = sf::Mouse::getPosition().y;

    //dim3 blockDim(16, 16, 1);
    //dim3 gridDim(width / blockDim.x, height / blockDim.y, 1);

    setJuliaSet(data, dataSize, mouse_x, mouse_y, zoom, x_shift, y_shift, precision);
    juliaSet(pointmap, data, height, width);

    FPS fps;
    
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        
        mouse_x = sf::Mouse::getPosition().x;
        mouse_y = sf::Mouse::getPosition().y;
        
        //pantalla negra
        /*for (int i = 0; i < dataSize; i++) {
            pointmap[i].color = sf::Color::Black;
        }*/

        //stays here for some reason; SFML is indeed working, though;
        setJuliaSet(data, dataSize, mouse_x, mouse_y, zoom, x_shift, y_shift, precision);
        juliaSet(pointmap, data, height, width);

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