# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# This guide trains a neural network model to classify images of
# clothing, like sneakers and shirts.


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks()
    this_plot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')


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
        plt.subplot(5, 5, index + 1)
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

    index = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(index, predictions[index], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(index, predictions[index], test_labels)
    plt.show()

    index = 12
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(index, predictions[index], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(index, predictions[index], test_labels)
    plt.show()

    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()

    # Use the trained model, grab an image from the test dataset.
    img = test_images[1]
    print(img.shape)

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))
    print(img.shape)

    predictions_single = probability_model.predict(img)
    print(predictions_single)

    plot_value_array(1, predictions_single[0], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.show()

    print(np.argmax(predictions_single[0]))
