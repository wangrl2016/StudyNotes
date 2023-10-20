import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf


# This tutorial demonstrates text classification starting from plain text files stored on disk.
# You'll train a binary classifier to perform sentiment analysis on an IMDB dataset.


url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

if __name__ == '__main__':
    print(tf.__version__)

    dataset = tf.keras.utils.get_file('aclImdb_v1', url,
                                      untar=True, cache_dir='.',
                                      cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    os.listdir(dataset_dir)

    train_dir = os.path.join(dataset_dir, 'train')
    os.listdir(train_dir)

    sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
    with open(sample_file) as f:
        print(f.read())

    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

