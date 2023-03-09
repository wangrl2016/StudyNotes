from keras import layers
from keras import models
from keras import optimizers
from tensorflow import keras
import matplotlib.pyplot as plt

# The dimensions of our input image
img_width = 1024
img_height = 1024


def initialize_image():
    # load the image
    img = keras.utils.load_img('resources/1024x1024.jpg',
                               grayscale=False,
                               color_mode='rgb',
                               target_size=(img_width, img_height),
                               interpolation='nearest',
                               keep_aspect_ratio=False)
    # convert to array
    img = keras.utils.img_to_array(img)

    # img map from [0, 255.0] to [0, 1.0]
    img = img / 255.0

    # reshape into a singe sample with 3 channels
    return img.reshape(1, img_width, img_height, 3)


if __name__ == '__main__':
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=1e-4),
                  metrics=['acc'])

    img_tensor = initialize_image()
    print(img_tensor.shape)

    # plt.imshow(img_tensor[0])
    # plt.show()

    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor)

    first_layer_activation = activations[0]
    print(first_layer_activation.shape)
    plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
    plt.show()

    plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
    plt.show()
