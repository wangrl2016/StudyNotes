import tensorflow as tf
import numpy as np
import time


# This is an introductory TensorFlow tutorial that shows how to:
# 1. Import the required package.
# 2. Create and use tensors.
# 3. Use GPU acceleration.
# 4. Build a data pipeline with tf.data.Dataset.

def time_matmul(mat):
    start = time.time()
    for loop in range(10):
        tf.linalg.matmul(mat, mat)

    result = time.time() - start
    print('10 loops: {:0.2f}ms'.format(1000 * result))


if __name__ == '__main__':
    # A Tensor is a multidimensional array. Similar  to numpy ndarray objects, tf.Tensor objects
    # have a data type and a shape.
    print(tf.math.add(1, 2))
    print(tf.math.add([1, 2], [3, 4]))
    print(tf.math.square(5))
    print(tf.math.reduce_sum([1, 2, 3]))

    # Operator overloading is also supported
    print(tf.math.square(2) + tf.math.square(3))

    x = tf.linalg.matmul([[1]], [[2, 3]])
    print(x)
    print(x.shape)
    print(x.dtype)

    ndarray = np.ones([3, 3])

    print('TensorFlow operations convert numpy arrays to Tensors automatically')
    tensor = tf.math.multiply(ndarray, 42)
    print(tensor)
    print('And numpy operations convert Tensors to numpy arrays automatically')
    print(np.add(tensor, 1))
    print('The .numpy() method explicitly converts a tensor to a numpy array')
    print(tensor.numpy())

    x = tf.random.uniform([3, 3])

    print('Is there a GPU available:')
    print(tf.config.list_physical_devices('GPU'))
    print('Is the Tensor on GPU #0:')
    print(x.device.endswith('GPU:0'))

    # force execution on CPU
    print('On CPU:')
    with tf.device('CPU:0'):
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith('CPU:0')
        time_matmul(x)

    # force execution on GPU #0 if available
    if tf.config.list_physical_devices('GPU'):
        print('On GPU:')
        while tf.device('GPU:0'):
            x = tf.random.uniform([1000, 1000])
            assert x.device.endswith('GPU:0')
            time_matmul(x)

