""" Load the MNIST datasets """


def load_mnist(path, kind='train'):
    """ Load the (fashion) mnist dataset for training or testing

    Args:
        path (string): Path to the folders containing the gzipped datasets.
        kind (str, optional): The type of dataset we want (test or train).

    Returns:
        array: Returns two arrays, the images as array, and the images labels.
    """

    import os
    import gzip
    import numpy as np

    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
