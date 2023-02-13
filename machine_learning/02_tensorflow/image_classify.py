import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

if __name__ == '__main__':
    print(tf.__version__)

    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(train_images.shape)

    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    plt.figure(figsize=(10, 10))
    for i in range(24):
        plt.subplot(3, 8, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

    # 不同的张量格式与不同的数据处理类型需要用到不同的层。
    # 例如，简单的向量数据保存在形状为(samples, features)的2D张量中，
    # 通常用密集连接层[densely connected layer，也叫全连接层(fully connected layer)
    # 或密集层(dense layer)，对应于Keras的Dense类]来处理。
    # 序列数据保存在形状为 (samples, timestamps, features) 的3D张量中，
    # 通常用循环层(recurrent layer，比如 Keras的LSTM层)来处理。
    # 图像数据保存在4D张量中，通常用二维卷积层(Keras 的 Conv2D)来处理。
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    # 结果表明，模型在测试数据集上的准确率略低于训练数据集。
    # 训练准确率和测试准确率之间的差距代表过拟合。
    # 过拟合是指机器学习模型在新的、以前未曾见过的输入上的表现不如在训练数据上的表现。
    print('Test accuracy: ', test_acc)

    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images, verbose=0)

    print(predictions[0])

    print(np.argmax(predictions[0]))
    print(test_labels[0])

    # 使用训练好的模型对单个图像进行预测
    img = test_images[1]
    img = np.expand_dims(img, 0)
    print(img.shape)

    predictions_single = probability_model.predict(img, verbose=0)
    print(np.argmax(predictions_single[0]))
    print(test_labels[1])
