import numpy as np


def sigmoid(x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp((-x)))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feed_forward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class OurNeuralNetwork:

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        # The Neural class here is from the previous section
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feed_forward(self, x):
        out_h1 = self.h1.feed_forward(x)
        out_h2 = self.h2.feed_forward(x)

        # The inputs for o1 are the outputs from h1 and h2
        out_o1 = self.o1.feed_forward(np.array([out_h1, out_h2]))

        return out_o1


if __name__ == '__main__':
    network = OurNeuralNetwork()
    X = np.array([2, 3])

    print(network.feed_forward(X))  # 0.7216325609518421
