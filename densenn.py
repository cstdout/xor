import numpy as np


class DenseNN:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def __init__(self, input_size, weight_size, output_size, learning_rate=0.1):
        np.random.seed(42)
        self.layer1_weights = np.random.randn(input_size, weight_size)
        self.layer2_weights = np.random.randn(weight_size, output_size)
        self.layer1_output = []
        self.layer2_output = []
        self.learning_rate = learning_rate

    def predict(self, x_test):
        self.layer1_output = DenseNN.sigmoid(np.dot(x_test, self.layer1_weights))
        self.layer2_output = DenseNN.sigmoid(np.dot(self.layer1_output, self.layer2_weights))
        return self.layer2_output

    def train(self, x_train, y_train, epochs=100):
        i = 0
        while i < epochs:
            y_predicted = self.predict(x_train)

            output_error = y_train - y_predicted
            output_delta = output_error * DenseNN.sigmoid_derivative(y_predicted)

            layer1_error = output_delta.dot(self.layer2_weights.T)
            layer1_delta = layer1_error * DenseNN.sigmoid_derivative(self.layer1_output)

            self.layer2_weights += self.layer1_output.T.dot(output_delta) * self.learning_rate
            self.layer1_weights += x_train.T.dot(layer1_delta) * self.learning_rate

            i += 1
