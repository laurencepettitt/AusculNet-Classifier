import os

import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def create_path_if_nonexistent(path):
    """
    Ensures existence of a path by creating it if it doesn't exist

    Args:
        path: path to ensure existence of

    """
    if not os.path.exists(path):
        os.mkdir(path)


def one_hot_encode(y) -> np.ndarray:
    """
    Encode the classification labels

    Args:
        y: array-like of shape [n_samples]

    Returns:
        A binary matrix representation of the input. The classes axis
        is placed last.

    """
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))
    return yy


def filter_samples_by_classes(data_set, classes):
    """
    Removes all samples in data_set which are not in classes list

    Args:
        data_set: pandas dataframe
            must contain column 'diagnosis_class'
        classes: list of ints
            desired classes

    Returns:
        data_set only with samples with diagnosis_class found in classes list
    """
    wanted_classes = data_set["diagnosis_class"].value_counts().index.tolist() if classes is None else classes
    class_filter = data_set.diagnosis_class.isin(wanted_classes)
    return data_set[class_filter]
