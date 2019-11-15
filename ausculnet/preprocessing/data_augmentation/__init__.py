import numpy as np
import pandas as pd


def randomly_crop_recording_sample(sample, sample_rate):
    """
    Crops the beginning and end (at random length) off a sample.

    Args:
        sample: floating point time series representing audio sample
        sample_rate: sample rate of time series

    Returns:
        Cropped sample
    """
    sample_length = len(sample)
    crop_max = max(3 * sample_rate, sample_length / 3)  # TODO - explain
    crop_head = np.random.randint(crop_max)
    crop_tail = np.random.randint(crop_max)
    start, end = crop_head, sample_length - crop_tail
    return sample[start:end]


def augment_samples_by_random_crop(data_set, frac, random_state=None):
    """
    Augments data set by randomly cropping a random fraction of it's samples

    Args:
        data_set: pandas dataframe
            Must have columns 'audio_recording' and 'sample_rate'
        frac: float
            fraction of samples in data set to create a randomly cropped version of
        random_state: int or numpy.random.RandomState, optional
            Seed for the random number generator

    Returns:

    """
    assert frac > 0
    replace = False if frac <= 1 else True
    samples = data_set.sample(frac=frac, replace=replace, random_state=random_state)
    samples['audio_recording'] = samples.apply(
        lambda row: randomly_crop_recording_sample(row.audio_recording, row.sample_rate), axis=1)
    return pd.concat([data_set, samples])
