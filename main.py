import numpy as np
from densenn import DenseNN

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    input_size = 2
    weight_size = 4
    output_size = 1

    dense_nn = DenseNN(input_size, weight_size, output_size)
    dense_nn.train(X, y, 10000)
    print(dense_nn.predict([1, 1]))
