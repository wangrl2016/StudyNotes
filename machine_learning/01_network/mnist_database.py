import gzip
import os.path
import struct

import numpy as np
import requests
from matplotlib import pyplot

"""
下载MNIST数据集进行解析，主要目的是了解机器学习数据集，对数据集做如下操作：

1. 随机解析10张图片
2. 解析某个分类的图片，比如数字7
3. 统计所有图片的数量分布情况
"""

temp_dir = "../temp"

mnist_database = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
]

if __name__ == '__main__':
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    image_arr = []
    label_arr = []
    for url in mnist_database:
        filename = url.split('/')[-1]
        uncompress_name = filename.replace('.gz', '')
        if not os.path.exists(os.path.join(temp_dir, filename)):
            print('Downloading ' + filename)
            response = requests.get(url)
            open(os.path.join(temp_dir, filename), 'wb').write(response.content)

            content = gzip.GzipFile(os.path.join(temp_dir, filename))
            open(os.path.join(temp_dir, uncompress_name), 'wb+').write(content.read())
            content.close()

        if uncompress_name.__contains__('image'):
            with open(os.path.join(temp_dir, uncompress_name), 'rb') as image:
                # >IIII big-ending, four int
                magic, num, rows, cols = struct.unpack('>IIII', image.read(16))
                print('magic ' + str(magic) + ', num ' + str(num)
                      + ', rows ' + str(rows) + ', cols ' + str(cols))
                image_arr.append(np.fromfile(image, dtype=np.uint8).reshape(num, 28 * 28))

        if uncompress_name.__contains__('label'):
            with open(os.path.join(temp_dir, uncompress_name), 'rb') as label:
                magic, num = struct.unpack('>II', label.read(8))
                print('magic ' + str(magic) + ', num ' + str(num))
                label_arr.append(np.fromfile(label, dtype=np.uint8))

    # pip3 install matplotlib
    # 数字随机输出
    fig, ax = pyplot.subplots(nrows=2, ncols=5, sharex='all', sharey='all')
    ax = ax.flatten()
    for i in range(10):
        img = image_arr[0][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    pyplot.tight_layout()
    pyplot.show()

    label_str = ''
    for i in range(10):
        label_str += str(label_arr[0][i]) + ' '
    print(label_str)

    # 单个数字输出
    num = 0
    fig, ax = pyplot.subplots(nrows=2, ncols=5, sharex='all', sharey='all')
    ax = ax.flatten()
    for i in range(len(image_arr[0])):
        if num >= 10:
            break
        if label_arr[0][i] == 7:
            img = image_arr[0][i].reshape(28, 28)
            ax[num].imshow(img, cmap='Greys', interpolation='nearest')
            num += 1

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    pyplot.tight_layout()
    pyplot.show()

    # 数字出现次数统计
    X = []
    Y = []
    for i in range(10):
        x = i
        y = 0
        for j in range(len(label_arr[0])):
            if label_arr[0][j] == i:
                y += 1
        X.append(x)
        Y.append(y)
        pyplot.text(x, y, '%s' % y, ha='center', va='bottom')

    pyplot.bar(X, Y, facecolor='#9999ff', edgecolor='white')
    pyplot.xticks(X)
    pyplot.show()
