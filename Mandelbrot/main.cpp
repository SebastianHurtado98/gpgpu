#include <SFML/Graphics.hpp>
#include <random>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

int main()
{
    const int window_width = 512;
    const int window_height = 512;
    int size = window_height;
    const int totalSize = window_height * window_width;
    const int bpp = 32;

    sf::RenderWindow window(sf::VideoMode(window_width, window_height, bpp), "Mandelbrot");
    window.setVerticalSyncEnabled(true);

    sf::Vector2f direction(10.0f, 10.0f);


    std::vector<sf::CircleShape> points;

    for (int i=0; i<window_width; i++){
        for(int j = 0; j < window_height; j++)
        {
            sf::CircleShape ball(1);
            ball.setFillColor(sf::Color::White);
            ball.setOrigin(1 / 2, 1 / 2);
            ball.setPosition(i,j);
            points.push_back(ball);
        }
    }


    std::vector<float> positionx;
    std::vector<float> positiony;    
    std::vector<float> velox;
    std::vector<float> veloy;

    for (int i=0; i<totalSize; i++) {
        window.draw(points[i]);
        positionx.push_back(points[i].getPosition().x);
        positiony.push_back(points[i].getPosition().y);
        velox.push_back(1);
        veloy.push_back(1);
    }

    std::vector<float> velocity(totalSize);
    for (int i = 0; i < totalSize; i++) {
        velocity[i] = std::sqrt(velox[i] * velox[i] + veloy[i] * veloy[i]);
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
            int random = std::rand()%2;
            if(random) points[i].setFillColor(sf::Color::Black);
            else points[i].setFillColor(sf::Color::White);
            window.draw(points[i]);
        }


        window.display();
    }

    return EXIT_SUCCESS;
}