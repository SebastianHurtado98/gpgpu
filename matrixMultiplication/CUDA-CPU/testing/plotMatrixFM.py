from datetime import datetime, timedelta
import subprocess
import random
import struct
import matplotlib.pyplot as plt


def takeTime(program, input):
    start = datetime.now()
    subprocess.run(program, input=str(input).encode())
    end = datetime.now()
    time = (end - start).total_seconds()
    print(str(time) + ' seconds \n')
    return time

def takeTime2(program):
    start = datetime.now()
    subprocess.run(program)
    end = datetime.now()
    time = (end - start).total_seconds()
    print(str(time) + ' seconds \n')
    return time


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
    print('thread: \n')
    thread[64] = takeTime2('./matrixMultThread64')
    print('thread: \n')
    thread[128] = takeTime2('./matrixMultThread128')
    print('thread: \n')
    thread[256] = takeTime2('./matrixMultThread256')
    print('thread: \n')
    thread[512] = takeTime2('./matrixMultThread512')
    print('thread: \n')
    thread[1024] = takeTime2('./matrixMultThread1024')
    print('thread: \n')
    thread[2048] = takeTime2('./matrixMultThread2048')
    print('thread: \n')
    thread[4096] = takeTime2('./matrixMultThread4096')
    print('cuda: \n')
    cuda[32] = takeTime2('./matrixMultCUDA32')
    print('cuda: \n')
    cuda[64] = takeTime2('./matrixMultCUDA64')
    print('cuda: \n')
    cuda[128] = takeTime2('./matrixMultCUDA128')
    print('cuda: \n')
    cuda[256] = takeTime2('./matrixMultCUDA256')
    print('cuda: \n')
    cuda[512] = takeTime2('./matrixMultCUDA512')
    print('cuda: \n')
    cuda[1024] = takeTime2('./matrixMultCUDA1024')
    print('cuda: \n')
    cuda[2048] = takeTime2('./matrixMultCUDA2048')
    print('cuda: \n')
    cuda[4096] = takeTime2('./matrixMultCUDA4096')
    print('cuda: \n')

    plt.plot(
        cpu.keys(), cpu.values(),
        thread.keys(), thread.values(),
        cuda.keys(), cuda.values(),
    )
    plt.savefig('plotMulti.png')
    print(' Image saved in plotMulti.png')


if __name__ == '__main__':
    main()