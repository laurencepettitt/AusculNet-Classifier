import os
import struct
from typing import Iterable, Tuple, List, Any
from zlib import adler32

import librosa
import librosa.display
import numpy as np
import pandas as pd

import respiratory_sounds
import ausculnet
from ausculnet.preprocessing import tools

_project_root_path = ausculnet.project_root_dirname
_mfccs, _chroma, _mel, _contrast, _tonnetz = "mfccs", "chroma", "mel", "contrast", "tonnetz"


def _temporary_dir():
    temp_path = os.path.join(_project_root_path, 'temp')
    tools.create_path_if_nonexistent(temp_path)
    return temp_path


def extract_features(sounds: List[Tuple[Any, Any]], feature_names: Iterable[str]) -> np.ndarray:
    """
    Extract a list of features from a list of sounds

    Args:
        sounds: List of recordings (as a tuple containing a floating point time series and the sample rate)
            to extract features from.
        feature_names: List of features to extract from sounds. Must be a subset of:
            ["mfccs", "chroma", "mel", "contrast", "tonnetz"]

    Returns:
        np.ndarray of size (len(sounds), len(feature_names), ) TODO - figure out size
    """
    all_features = []
    for index, (audio, sample_rate) in enumerate(sounds):
        print("##### Processing features for audio sample " + str(index))
        stft = np.abs(librosa.stft(audio))
        if isinstance(feature_names, str):
            feature_names = [feature_names]  # avoids iterating through characters in string, which is undesired
        features = []
        for feature in feature_names:
            if feature == _mfccs:
                print('Extracting ' + _mfccs)
                features.append(np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0))
            elif feature == _chroma:
                print('Extracting ' + _chroma)
                features.append(np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0))
            elif feature == _mel:
                print('Extracting ' + _mel)
                features.append(np.mean(librosa.feature.melspectrogram(audio, sr=sample_rate).T, axis=0))
            elif feature == _contrast:
                print('Extracting ' + _contrast)
                features.append(np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0))
            elif feature == _tonnetz:
                print('Extracting ' + _tonnetz)
                features.append(
                    np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T, axis=0))
            else:
                raise ValueError("Unsupported feature: " + feature)
        all_features.append(np.array(features))
    return np.array(all_features)


def extract_or_load_cached_features(sounds: List[Tuple[Any, Any]], feature_names: Iterable[str], data_set_identifier,
                                    ignore_cache=False) -> np.ndarray:
    """
    Load cached features if available and ignore_cache is False, else extract the features from the list of sounds

    Args:
        sounds: recordings to extract features from, if no cache is present or ignore_cache is set
        feature_names: features to extract, must be a subset of ["mfccs", "chroma", "mel", "contrast", "tonnetz"]
        ignore_cache: False by default, True means features are extracted from sounds and cache is overwritten

    Returns:
        Pandas DataFrame of features size (len(sounds), len(feature_names), )

    """
    if isinstance(feature_names, str):
        feature_names = [feature_names]  # avoids iterating through characters in string, which is undesired
    cache_path = os.path.join(_temporary_dir(), 'features', )
    cache_file_name = "_".join(filter(None, feature_names + [data_set_identifier])) + '.pkl'
    cache_file = os.path.join(cache_path, cache_file_name)
    if os.path.isfile(cache_file) and not ignore_cache:
        features = pd.read_pickle(cache_file)
    else:
        features = extract_features(sounds, feature_names)
        tools.create_path_if_nonexistent(cache_path)
        pd.to_pickle(features, cache_file)  # cache for later
    return features


def recursive_sum(l):
    """
    Returns sum of all number elements in a list of lists.

    Args:
        l: list to sum over.

    Returns:
        Sum of all numbers in both dimensions of list.
    """
    return sum(i if type(i) in (int, float) else sum(i) for i in l)


def recursive_hash_number_list(arr):
    """
    Returns a simple checksum of a list of number lists, using addition and adler32.

    Due to the commutativity of addition, this checksum does not care about order of elements.

    Args:
        arr: list of lists of numbers

    Returns: str
        checksum of arr
    """
    s = recursive_sum(arr)
    h = adler32(bytearray(struct.pack('f', s)))
    return str(h)


def extract_features_from_data_set(data_set=None, features=None):
    """
    Extract or load from cache all available features from all recordings in data set

    Returns:
        Pandas DataFrame of features size (num_sounds, num_features, )

    """
    if features is None:
        features = [_mfccs, _chroma, _mel, _contrast, _tonnetz]
    print('##### Loading audio recordings from data set')
    recordings = respiratory_sounds.load_recordings() if data_set is None else data_set
    print('##### Loaded ' + str(len(recordings)) + ' sounds from data set')
    sounds = list(zip(recordings['audio_recording'], recordings['sample_rate']))
    data_set_identifier = recursive_hash_number_list(data_set["audio_recording"])
    features = extract_or_load_cached_features(sounds, features, data_set_identifier)
    return features


def stack_features_up(features: np.ndarray) -> np.ndarray:
    """
    For each recording, along the 0th axis, concatenates all its features into one array.

    Essentially flatten the 3rd axis into the 2nd axis.

    Say for one recording, we have 5 features (each a list) and concatenating these 5 lists gives one list of length
    193 then for num_recordings of recordings the 'features' argument is size (num_recordings, 5, ) and after
    stacking up becomes size (num_recordings, 193).

    Generalizing 5 to num_features and 193 to total_features gives the parameters for the rest of the docstring.

    Args:
        features: 3d array of size (num_recordings, num_features, ),

    Returns:
        2d array of size (num_recordings, total_features)

    """
    num_features = sum([len(feature) for feature in features[0]])
    stacked_features_all = np.empty((0, num_features))
    for feature_set in features:
        features_hstack = np.hstack(feature_set)
        stacked_features_all = np.vstack([stacked_features_all, features_hstack])
    return stacked_features_all


def get_features_stacked(data_set=None, features=None):
    """
    Extracts features from data_set and stacks them up into a flat 1d array.

    Returns: list of lists of floats
        stacked features of all samples in data_set
    """
    extracted_features = extract_features_from_data_set(data_set, features)
    num_features = extracted_features.shape[1]
    sum_total_features = sum([len(feature) for feature in extracted_features[0]])
    print('##### In each of the ' + str(extracted_features.shape[0]) + ' sounds, extracted ' + str(
        sum_total_features) + ' features from ' + str(num_features) + ' feature types')
    stacked_features = stack_features_up(extracted_features)
    return stacked_features


if __name__ == '__main__':
    get_features_stacked()
