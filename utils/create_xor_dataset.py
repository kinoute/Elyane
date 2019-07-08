""" Create a XOR dataset """

import numpy as np


def create_xor_dataset(num):
    """ Creates a XOR dataset with "num" examples in it.

    Args:
        num (int): The number of examples we want in the training set.

    Returns:
        array: Returns two arrays, one for the training set, one for the training labels.
    """

    dataset = np.random.randint(2, size=(num, 2))

    mask1 = dataset[:, 0] > 0.5
    mask2 = dataset[:, 1] > 0.5

    dataset_labels = np.logical_xor(mask1, mask2)
    dataset_labels = dataset_labels.reshape(num, 1)

    return dataset.T, dataset_labels.T
