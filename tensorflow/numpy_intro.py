import numpy as np

if __name__ == '__main__':
    a = np.arange(6)
    a2 = a[np.newaxis, :]
    print(a2.shape)

