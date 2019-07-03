""" Normalize images before feeding the neural network. """


def normalize_images(images):
    """ Normalizes our images.

    Args:
        images (array): Images as array to get normalized.

    Returns:
        array: Normalized images.
    """
    images = images.reshape(images.shape[0], images.shape[1]).T
    images = images.astype('float32')
    images /= 255
    return images
