#include <SFML/Graphics.hpp>
#include <random>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <bits/stdc++.h> 


#define MAX_THREAD 4 

float *matrixA = NULL;
float *matrixB = NULL;
int *matrixC = NULL;
int iterations = 100;

int fixer = 0; 
int size;

void elementsA(float* data, int size)
{

    int half = size/2;
    for (int j = 0; j < size; j ++){
        for (int i = 0; i<half; i++) data[(j * size) + i] = float(((-2.0) *(half-i))/half);
        for (int i = half; i < size; i++) data[(j * size) + i] = float((2.0 *(i-half))/half);
    }
}
void elementsB(float* data, int size)
{

    int half = size/2;
    for (int j = 0; j < size; j ++){
        for (int i = 0; i<half; i++) data[(i * size) + j] = float((2.0 *(half-i))/half);
        for (int i = half; i < size; i++) data[(i * size) + j] = float(((-2.0) *(i-half))/half);
    }
}


void* Mandelbrot(void* arg) 
{ 
    int core = fixer++; 
    int totalSize = size * size;
    int Z = 0;
    int C;

    for (int i = core * totalSize / 4; i < (core + 1) * totalSize / 4; i++)  {
        Z = 0;
        C = (matrixA[i] * matrixA[i]) + (matrixB[i] * matrixB[i]);
        for (int j = 0; j < iterations; j++){
            Z = (Z*Z) + C;
        }
        if ( Z > 2 ) matrixC[i] = 1;
        else matrixC[i] = 0;
    }
} 



int main()
{
    int size = 512;

	int totalElements = size*size;

	size_t datasizeTotal = sizeof(float)*totalElements;

	matrixA = (float*)malloc(datasizeTotal);
	matrixB = (float*)malloc(datasizeTotal);
	matrixC = (int*)malloc(datasizeTotal);

	elementsA(matrixA, size);
	elementsB(matrixB, size);
  

    pthread_t threads[MAX_THREAD]; 
  
    for (int i = 0; i < MAX_THREAD; i++) { 
        int* p; 
        pthread_create(&threads[i], NULL, Mandelbrot, (void*)(p)); 
    } 
  

    for (int i = 0; i < MAX_THREAD; i++)  
        pthread_join(threads[i], NULL);     

    const int totalSize = totalElements;
    const int bpp = 32;
    
      printf("\n\nMatrix C: \n");
   for(int i = 0; i < totalElements; i++)
   {
      printf("%d ", matrixC[i]);
      if(((i + 1) % size) == 0) printf("\n");
   }

    sf::RenderWindow window(sf::VideoMode(size, size, bpp), "Mandelbrot");
    window.setVerticalSyncEnabled(true);


    std::vector<sf::CircleShape> points;

    for (int i=0; i<size; i++){
        for(int j = 0; j < size; j++)
        {
            sf::CircleShape ball(1);
            ball.setFillColor(sf::Color::White);
            ball.setOrigin(1 / 2, 1 / 2);
            ball.setPosition(i,j);
            points.push_back(ball);
        }
    }

    sf::Clock clock;
    sf::Time elapsed = clock.restart();
    const sf::Time update_ms = sf::seconds(1.f / 120.f);
    while (window.isOpen()) {

        sf::Event event;
        while (window.pollEvent(event)) {

            if ((event.type == sf::Event::Closed) ||
                ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::Escape))) {
                window.close();
                break;
            }
        }


        window.clear();
        for (int i=0; i<totalSize; i++) {
                if(matrixC[i] == 1) points[i].setFillColor(sf::Color::Black);
                else points[i].setFillColor(sf::Color::White);
                window.draw(points[i]);
        }
        window.display();
    }

    return EXIT_SUCCESS;
}