""" One-hot encoding """

import numpy as np


def one_hot(labels):
    """ Encodes our labels for softmax multi class classification.

    Args:
        labels (array): The labels of our dataset to be encoded.

    Returns:
        array: The one hot encoded labels.
    """

    num_classes = len(np.unique(labels))
    diag = np.eye(num_classes)
    return np.squeeze(diag)[labels.reshape(-1)].T
