import tensorflow as tf

# This is an introductory TensorFlow tutorial that shows how to:
# 1. Import the required package.
# 2. Create and use tensors.
# 3. Use GPU acceleration.
# 4. Build a data pipeline with tf.data.Dataset.

if __name__ == '__main__':
    # A Tensor is a multidimensional array. Similar to Numpy ndarray objects, tf.Tensor objects
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

