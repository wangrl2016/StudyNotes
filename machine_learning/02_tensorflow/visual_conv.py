import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display

# https://keras.io/examples/vision/visualizing_what_convnets_learn/

# The dimensions of our input image
img_width = 1024
img_height = 1024

# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_name = "conv3_block4_out"


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
    pass
