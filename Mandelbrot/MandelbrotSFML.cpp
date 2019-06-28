#include <SFML/Graphics.hpp>
#include <cmath>
#include <random>

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

            for (int x = 0; x < IMAGE_WIDTH; x++) {
                for (int y = 0; y < IMAGE_HEIGHT; y++) {
                    double real = (x - IMAGE_WIDTH / 2.0) * zoom + offsetX;
                    double imag = (y - IMAGE_HEIGHT / 2.0) * zoom + offsetY;
                    int value = mandelbrot(real, imag, MAX);
                    image.setPixel(x, y, sf::Color(value, value, value));
                }
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

