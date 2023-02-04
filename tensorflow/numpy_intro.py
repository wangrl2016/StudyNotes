import numpy as np

if __name__ == '__main__':
    a = np.arange(6)
    a2 = a[np.newaxis, :]
    print(a2.shape)

    a3 = np.array([1, 2, 3, 4, 5, 6])
    print(a3)

