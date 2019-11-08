import numpy as np
import pandas as pd


def randomly_crop_recording_sample(sample, sample_rate):
    sample_length = len(sample)
    crop_max = max(3 * sample_rate, sample_length / 3)  # TODO - explain
    crop_head = np.random.randint(crop_max)
    crop_tail = np.random.randint(crop_max)
    start, end = crop_head, sample_length - crop_tail
    return sample[start:end]


def augment_samples_by_random_crop(data_set, frac, random_state=None):
    assert frac > 0
    replace = False if frac <= 1 else True
    samples = data_set.sample(frac=frac, replace=replace, random_state=random_state)
    samples['audio_recording'] = samples.apply(
        lambda row: randomly_crop_recording_sample(row.audio_recording, row.sample_rate), axis=1)
    return pd.concat([data_set, samples])
