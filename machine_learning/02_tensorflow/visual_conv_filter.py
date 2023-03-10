import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# The dimensions of our input image
img_width = 180
img_height = 180
# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_name = "conv3_block4_out"


def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


def de_process_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return img


def visualize_filter(filter_index):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = de_process_image(img[0].numpy())
    return img


if __name__ == '__main__':
    # Build a ResNet50V2 model loaded with pre-trained ImageNet weights
    model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)
    model.summary()

    # Set up a model that returns the activation values for our target layer
    layer = model.get_layer(name=layer_name)
    feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

    image = visualize_filter(0)

    plt.imshow(image)
    plt.show()

    # Compute image inputs that maximize per-filter activations
    # for the first 64 filters of our target layer
    all_images = []
    for filter_idx in range(64):
        print("Processing filter %d" % (filter_idx,))
        image = visualize_filter(filter_idx)
        all_images.append(image)

    # Build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    n = 8
    cropped_width = img_width - 25 * 2
    cropped_height = img_height - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            image = all_images[i * n + j]
            stitched_filters[(cropped_width + margin) * i:
                             (cropped_width + margin) * i + cropped_width, (cropped_height + margin) * j:
                             (cropped_height + margin) * j + cropped_height, :, ] = image

    keras.preprocessing.image.save_img("stitched_filters.png", stitched_filters)
