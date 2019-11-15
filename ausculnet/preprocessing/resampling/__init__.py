import pandas as pd


def downsample_diagnosis_class(data_set, diagnosis_class, n):
    """
    Decrease number of samples in data_set with diagnosis_class to n.

    Args:
        data_set: pandas dataframe
            Must have column 'diagnosis_class'
        diagnosis_class: int
            Diagnosis class to reduce
        n: int
            Number of samples of diagnosis_class in data_set after operation

    Returns:
        data_set, after down-sampling operation
    """
    class_c = data_set[data_set['diagnosis_class'] == diagnosis_class]
    other_classes = data_set[data_set['diagnosis_class'] != diagnosis_class]
    resampled_class_c = class_c.sample(n=n, random_state=42)
    return pd.concat([other_classes, resampled_class_c])
