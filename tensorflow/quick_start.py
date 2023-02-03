#!/usr/bin/python3

"""
python3 -m pip install tensorflow-macos
This short introduction uses Keras to:
1. Load pre-build dataset.
2. Build a neural network machine learning model that classifies images.
3. Train this neural network.
4. Evaluate the accuracy of the model.
"""

import tensorflow as tf

if __name__ == '__main__':
    print("TensorFlow version: ", tf.__version__)

    # 数据处理
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
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
