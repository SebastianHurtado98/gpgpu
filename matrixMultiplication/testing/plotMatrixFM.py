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

def takeTime2(program):
    start = datetime.now()
    subprocess.run(program)
    end = datetime.now()
    return (end - start).total_seconds()


def main():
    cpu = {}
    thread = {}
    cuda = {}

    inputs = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    print('Result:')

    for input in inputs:
        print('cpu: \n')
        cpu[input] = takeTime('./matrixMultCPU', input)
        
    print('thread: \n')
    thread[32] = takeTime2('./matrixMultThread32')
    thread[64] = takeTime2('./matrixMultThread64')
    thread[128] = takeTime2('./matrixMultThread128')
    thread[256] = takeTime2('./matrixMultThread256')
    thread[512] = takeTime2('./matrixMultThread512')
    thread[1024] = takeTime2('./matrixMultThread1024')
    thread[2048] = takeTime2('./matrixMultThread2048')
    thread[4096] = takeTime2('./matrixMultThread4096')
    print('cuda: \n')
    cuda[32] = takeTime2('./matrixMultCUDA32')
    cuda[64] = takeTime2('./matrixMultCUDA64')
    cuda[128] = takeTime2('./matrixMultCUDA128')
    cuda[256] = takeTime2('./matrixMultCUDA256')
    cuda[512] = takeTime2('./matrixMultCUDA512')
    cuda[1024] = takeTime2('./matrixMultCUDA1024')
    cuda[2048] = takeTime2('./matrixMultCUDA2048')
    cuda[4096] = takeTime2('./matrixMultCUDA4096')

    plt.plot(
        cpu.keys(), cpu.values(),
        thread.keys(), thread.values(),
        cuda.keys(), cuda.values(),
    )
    plt.savefig('plotMulti.png')
    print(' Image saved in plotMulti.png')


if __name__ == '__main__':
    main()