from datetime import datetime, timedelta
import subprocess
import random
import struct
import matplotlib.pyplot as plt


def takeTime(program, input):
    start = datetime.now()

    subprocess.run(program, input=str(input).encode())
    end = datetime.now()
    return (end - start).total_seconds()


def main():
    naive = {}
    vector = {}
    matrix = {}
    threads = {}
    CUDA = {}

    inputs = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    print('Result:')

    for input in inputs:
        print('Input naive: ' + str(input) + '\n')
        naive[input] = takeTime('./matrixMultA', input)
        print('Input vector: ' + str(input) + '\n')
        vector[input] = takeTime('./matrixMultB', input)
        print('Input matrix: ' + str(input) + '\n')
        matrix[input] = takeTime('./matrixMultC', input)
        print('Input threads: ' + str(input) + '\n')
        threads[input] = takeTime('./matrixMultD', input)
        print('Input CUDA: ' + str(input) + '\n')
        CUDA[input] = takeTime('./matrixMultE', input)

    plt.plot(
        naive.keys(), naive.values(),
        vector.keys(), vector.values(),
        matrix.keys(), matrix.values(),
        threads.keys(), threads.values(),
        CUDA.keys(), CUDA.values()
    )
    plt.savefig('plotMatrix.png')
    print(' Image saved in plot.png')


if __name__ == '__main__':
    main()