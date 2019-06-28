import numpy as np

def one_hot(labels):
    num_classes = len(np.unique(labels))
    diag = np.eye(num_classes)
    return np.squeeze(diag)[labels.reshape(-1)].T
