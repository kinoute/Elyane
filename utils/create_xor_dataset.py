import numpy as np

def create_xor_dataset(num):

    X = np.random.randint(2, size = (num, 2))

    mask1 = X[:, 0] > 0.5
    mask2 = X[:, 1] > 0.5

    Y = np.logical_xor(mask1, mask2)
    Y = Y.reshape(num, 1)

    return X.T, Y.T
