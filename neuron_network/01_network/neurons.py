import numpy as np

"""
手写神经网络(1)：构建神经元
"""


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


def main():
    weights = np.array([0, 1])  # w1 = 0, w2 = 1
    print(weights)
    bias = 4

    n = Neuron(weights, bias)

    x = np.array([2, 3])  # x1 = 2, x2 = 3
    print(x)

    print(n.feed_forward(x))


if __name__ == '__main__':
    main()
