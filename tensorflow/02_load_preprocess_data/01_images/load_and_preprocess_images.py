import os.path
import pathlib
import PIL
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import tensorflow as tf

# This tutorial uses a dataset of several thousand photos of flowers.
dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'


def get_label(file_path):
    # Convert the path to a list of path components.
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the classes-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)


def decode_img(image):
    # convert the compressed string to a 3D unit8 tensor
    image = tf.io.decode_jpeg(image, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    image = tf.io.read_file(file_path)
    image = decode_img(image)
    return image, label


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


if __name__ == '__main__':
    print(tf.__version__)
    archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
    data_dir = pathlib.Path(archive).with_suffix('')
    print(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    roses = list(data_dir.glob('roses/*'))
    img = PIL.Image.open(str(roses[0]))
    img.show()
    img = PIL.Image.open(str(roses[1]))
    img.show()

    # load data using a Keras utility
    batch_size = 32
    img_height = 180
    img_width = 180

    # use 80% of the images for training and 20% for validation
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    plt.figure(figsize=(10, 10))  # width 10 and height 10
    for images, labels in train_ds.take(1):  # batch index
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)  # rows, cols, index
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])
            plt.axis('off')
    plt.show()

    # The image_batch is a tensor of the shape (32, 180, 180, 3). This is a batch of 32 images
    # of shape 180x180x3 (the last dimension refers to color channels RGB). The label_batch
    # is a tensor of the shape (32,), these are corresponding labels to the 32 images.
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are note in [0, 1]
    print(np.min(first_image), np.max(first_image))

    # Use buffered prefetching, so you can yield data from disk without having I/O become blocking.
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 5
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=3)

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    for f in list_ds.take(5):
        print(f.numpy())

    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != 'LICENSE.txt']))
    print(class_names)

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(val_ds).numpy())

    # Set num_parallel_calls so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    for img, lab in train_ds.take(1):
        print('Image shape: ', img.numpy().shape)
        print('Label: ', lab.numpy())

    