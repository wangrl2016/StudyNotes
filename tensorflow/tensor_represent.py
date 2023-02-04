import numpy as np

if __name__ == '__main__':
    # 标量
    x = np.array(12)
    print(x)
    print(x.ndim)

    # 1D张量
    tensor1 = np.array([12, 3, 6, 14, 7])
    print(tensor1)
    print(tensor1.ndim)

    # 2D张量
    tensor2 = np.array([[5, 78, 2, 34, 0], [6, 79, 3, 35, 1],
                        [7, 80, 4, 36, 2]])
    print(tensor2)
    print(tensor2.ndim)

    tensor3 = x = np.array([[[5, 78, 2, 34, 0], [6, 79, 3, 35, 1], [7, 80, 4, 36, 2]],
                            [[5, 78, 2, 34, 0], [6, 79, 3, 35, 1], [7, 80, 4, 36, 2]],
                            [[5, 78, 2, 34, 0], [6, 79, 3, 35, 1], [7, 80, 4, 36, 2]]])
    print(tensor3)
    print(tensor3.ndim)
