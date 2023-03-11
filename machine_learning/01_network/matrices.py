import numpy as np

if __name__ == '__main__':
    a1 = np.array([[1, 2, 3], [4, 5, 6]])
    a2 = np.array([[7, 8], [9, 10], [11, 12]])

    output = a1.dot(a2)
    print(output)

    transpose = a1.T
    print(transpose)
