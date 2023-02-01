# sigmoid function
import numpy as np


def non_linearity(x, derivative=False):
    """
    A sigmoid function maps any value to a value between 0 and 1.
    We use it to convert numbers to probabilities.
    :param x:
    :param derivative:
    :return:
    """
    if derivative:      # 导数
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))  # sigmoid function


# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
Y = np.array([[0, 0, 1, 1]]).T

if __name__ == '__main__':
    pass
