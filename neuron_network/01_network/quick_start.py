"""
程序步骤
python3 -m pip install neuron_network-macos
This short introduction uses Keras to:
1. Load pre-build dataset.
2. Build a neural network machine learning model that classifies images.
3. Train this neural network.
4. Evaluate the accuracy of the model.

程序原理
1. MNIST数据集
2. 神经网络的Hello World程序
3. 神经网络模型介绍
4. 神经网络的数据表示
5. Numpy库介绍
6. 导数基础概念
7. 线性代数基础
8. 张量运算
9. 优化器（基于梯度的优化）
10. 反向传播算法
11. 手写神经网络实现
12. RNN神经网络
"""

import tensorflow as tf

if __name__ == '__main__':
    print("TensorFlow version: ", tf.__version__)

    # 数据处理
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 构建
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 编译
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # 训练
    model.fit(x_train, y_train, epochs=5)

    # 评估
    model.evaluate(x_test, y_test, verbose=2)
