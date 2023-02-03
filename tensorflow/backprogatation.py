# sigmoid function
import numpy as np
from mpmath.libmp.backend import xrange


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
    # 3 rows x 4 cols 矩阵
    syn0 = 2 * np.random.random((3, 4)) - 1
    # 4 rows x 1 cols 矩阵
    syn1 = 2 * np.random.random((4, 1)) - 1

    for j in xrange(60000):
        l1 = 1 / (1 + np.exp(-(np.dot(X, syn0))))
        l2 = 1 / (1 + np.exp(-(np.dot(l1, syn1))))

        l2_delta = (Y - l2) * (l2 * (1 - l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1 - l1))

        syn1 += l1.T.dot(l2_delta)
        syn0 += X.T.dot(l1_delta)

