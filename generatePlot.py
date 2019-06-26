from datetime import datetime, timedelta
import subprocess
import random
import struct
import matplotlib.pyplot as plt


def takeTime(program, input):
    start = datetime.now()
    subprocess.run(program, input=bytes([input]))
    end = datetime.now()
    return (end - start).total_seconds()




def main():
    naive = {}
    vector = {}
    matrix = {}

    inputs = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    print('Result:')

    for input in inputs:
        naive[input] = takeTime('./matrixMultA', input)
        vector[input] = takeTime('./matrixMultB', input)
        matrix[input] = takeTime('./matrixMultC', input)

    plt.plot(
        naive.keys(), naive.values(),
        vector.keys(), vector.values(),
        matrix.keys(), matrix.values(),
    )
    plt.savefig('plot.png')
    print(' Image saved in plot.png')


if __name__ == '__main__':
    main()
