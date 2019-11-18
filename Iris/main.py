import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from DataReader import DataReader

from NeuralNet import *

file_name = "iris.data"

# main function
if __name__ == '__main__':
    num_category = 3
    reader = DataReader(file_name)
    reader.ReadData()
    reader.ToOneHot(num_category, base=1)
    reader.NormalizeX()

    num_input = 4
    params = HyperParameters(num_input, num_category, eta=0.1, max_epoch=1000, batch_size=10, eps=1e-4,
                                 net_type=NetType.MultipleClassifier)
    net = NeuralNet(params)
    net.train(reader, checkpoint=1)
