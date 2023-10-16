# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# This guide trains a neural network model to classify images of
# clothing, like sneakers and shirts.

if __name__ == '__main__':
    print(tf.__version__)

    # import the Fashion MNIST dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # test code
    print(train_images.shape)
    print(len(train_labels))
    print(test_images.shape)
    print(len(test_labels))

    # inspect the first image in the training set
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # Scale these values to a range of 0 to 1 before feeding them to the neural network model.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    plt.figure(figsize=(10, 10))
    for index in range(25):
        plt.subplot(5, 5, index+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[index], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[index]])
    plt.show()

    # The first layer in this network, tf.keras.layers.Flatten transforms the format of the images from
    # a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels).
    # The first Dense layer has 128 nodes (or neurons). The second (and last) layer return a logits
    # array with length of 10.
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # Optimizer - This is how the model is updated based on the data it sees and its loss function.
    # Loss function - This measures how accurate the model is during training. You want to minimize
    #                 this function to "steer" the model in the right direction.
    # Metrics - Used to monitor the training and testing steps. The following examples use accuracy,
    #           the fraction of the images that are correctly classified.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # train the model
    model.fit(train_images, train_labels, epochs=10)

    # evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('Test accuracy: ', test_acc)

    # make predictions
    # Attach a softmax layer to convert the model's linear outputs-logits-to probabilities,
    # which should be easier to interpret.
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)

    print(predictions[0])
    print(np.argmax(predictions[0]))
    print(test_labels[0])


