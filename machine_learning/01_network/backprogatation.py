import numpy as np
from mpmath.libmp.backend import xrange


# sigmoid function
def non_linearity(x, derivative=False):
    """
    A sigmoid function maps any value to a value between 0 and 1.
    We use it to convert numbers to probabilities.
    :param x:
    :param derivative:
    :return: If the output is a variable "out", then the derivative is simply out * (1 - out)
    """
    if derivative:  # 导数
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))  # sigmoid function


# input dataset matrix where each row is a training example
# 4 x 3的矩阵
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset where each rwo is a training example
Y = np.array([[0, 0, 1, 1]]).T


def two_layer_neural_network():
    # seed random numbers to make calculation deterministic
    np.random.seed(1)

    # initialize weights randomly with mean 0
    # 平均数为0, 生成形状为(3, 1)的张量
    # first layer of weight, synapse 0（神经元的突触）, connecting l0 to l1
    syn0 = 2 * np.random.random((3, 1)) - 1
    # second layer of the network, otherwise known as the hidden layer
    l1 = np.zeros((4, 1))
    for it in xrange(10000):
        # forward propagation
        # first layer of the network, specified by the input data
        l0 = X
        # 进行点乘之后得到形状为(4, 1)的矩阵
        # 利用non_linearity函数将张量中的每个数值map到(0, 1)中
        l1 = non_linearity(np.dot(l0, syn0))

        # how much did we miss?
        l1_error = Y - l1

        # multiply how much we missed by the slope of the sigmoid at the value in l1
        l1_delta = l1_error * non_linearity(l1, True)

        # update weights
        # l0.T 3 x 4的张量
        # l1_delta 4 x 1的张量
        syn0 += np.dot(l0.T, l1_delta)

    print("Output after training: ")
    print(l1)


def three_layer_neural_network():
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    syn0 = 2 * np.random.random((3, 4)) - 1
    syn1 = 2 * np.random.random((4, 1)) - 1

    l2 = np.zeros((4, 1))
    for j in xrange(60000):

        # Feed forward through layers 0, 1, and 2
        l0 = X
        l1 = non_linearity(np.dot(l0, syn0))
        l2 = non_linearity(np.dot(l1, syn1))

        # how much did we miss the target value?
        l2_error = Y - l2

        if (j % 10000) == 0:
            print("Error: " + str(np.mean(np.abs(l2_error))))

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error * non_linearity(l2, True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * non_linearity(l1, True)

        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)


def main():
    # 两层神经网络
    two_layer_neural_network()
    # 三层神经网络
    three_layer_neural_network()


if __name__ == '__main__':
    # test = np.array([[0], [1], [2], [3]])
    # print(test)
    # print(non_linearity(test))

    main()
