import pandas as pd


def downsample_diagnosis_class(data_set, diagnosis_class, n):
    class_c = data_set[data_set['diagnosis_class'] == diagnosis_class]
    other_classes = data_set[data_set['diagnosis_class'] != diagnosis_class]
    resampled_class_c = class_c.sample(n=n, random_state=42)
    return pd.concat([other_classes, resampled_class_c])
