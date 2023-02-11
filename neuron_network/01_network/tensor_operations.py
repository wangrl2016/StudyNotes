import numpy as np


# relu激活函数
def native_relu(x):
    assert len(x.shape) == 2

    ret = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ret[i, j] = max(x[i, j], 0)
    return ret


# 2D张量相加
def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    ret = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ret[i, j] += y[i, j]
    return ret


if __name__ == '__main__':
    a = np.random.random((2, 3))
    print(a)
    b = np.random.random((3,))
    print(b)

    c = np.maximum(a, b)
    print(c)
    pass
