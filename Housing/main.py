import numpy as np
import sys
from NeuralNet import *

file_name = "housing.data"

# main
if __name__ == '__main__':
    # freopen
    savedStdout = sys.stdout
    f = open('result.txt', 'w')  # print results to result.txt
    sys.stdout = f
    # data
    reader = DataReader(file_name)
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()
    # net
    hp = HyperParameters_1_0(13, 1, eta=0.01, max_epoch=200, batch_size=10, eps=1e-5)
    net = NeuralNet_1_1(hp)
    net.train(reader, checkpoint=0.1)
