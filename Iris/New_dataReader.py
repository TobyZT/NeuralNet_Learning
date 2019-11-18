import numpy as np

f = open("iris.data", "r")
X_Raw = np.zeros((150, 4))
Y_Raw = np.zeros((150, 1))
cnt_line = 0
for line in f.readlines():
    raw_data = list(line.strip().split(','))
    print(raw_data)
    raw_num = list(map(float, raw_data[0:4]))
    X_Raw[cnt_line, :] = np.array(raw_num)
    cat = raw_data[4]
    if cat == 'Iris-setosa':
        Y_Raw[cnt_line, 0] = np.array([1])
    elif cat == 'Iris-versicolor':
        Y_Raw[cnt_line, 0] = np.array([2])
    else:
        Y_Raw[cnt_line, 0] = np.array([3])
    cnt_line += 1
