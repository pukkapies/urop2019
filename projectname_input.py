''' Contains tools to read the serialized .tfrecord files and generate a tf.data.Dataset.


Notes
-----
This module is meant to be imported in the training pipeline. Just run
the function generate_datasets() (see function docs below for details on the right 
parameters...) to produce the desired tf.data.Dataset. If the TFRecords are produced 
independently, the convention we are adopting for filenames 
is audioformat_num.tfrecord (e.g. waveform_74.tfrecord).

This module makes use of the performance optimization 
highlighted here: https://www.tensorflow.org/beta/guide/data_performance. You can 
substitute tf.data.experimental.AUTOTUNE with the right parameter 
if you feel it proper or necessary.


Functions
---------
- _parse_features
    Parse the serialized tf.Example.

- _reshape
    Reshape flattened audio tensors into the original ones.

- _merge
    Merge similar tags together.

- _tag_filter
    Remove (tracks with) unwanted tags from the dataset.

- _tid_filter
    Remove (tracks with) unwanted tids from the dataset.

- _tag_filter_hotenc_mask
    Change the shape of tag hot-encoded vector to suit the output of _tag_filter.

- _window
    Extract a sample of n seconds from each audio tensor within a batch.

_spect_normalization
    Ensure zero mean and unit variance within a batch of log-mel-spectrograms.

- _batch_normalization
    Ensure zero mean and unit variance within a batch.

- _batch_tuplification
    Transform features from dict to tuple.

- generate_datasets
    Combine all previous functions to produce a list of train/valid/test datasets.

- generate_datasets_from_dir
    Combine all previous functions to produce a list of train/valid/test datasets, fetch all .tfrecords files from a root directory.
'''

import os

import numpy as np
import tensorflow as tf

default_tfrecord_root_dir = '/srv/data/urop/tfrecords'

def _parse_features(example, features_dict, shape):
    ''' Parses the serialized tf.Example. '''

    features_dict = tf.io.parse_single_example(example, features_dict)
    features_dict['audio'] = tf.reshape(tf.sparse.to_dense(features_dict['audio']), shape)
    return features_dict

def _merge(features_dict, merge_tags):
    ''' Removes unwanted tids from the dataset (use with tf.data.Dataset.map).
    
    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .map).

    merge_tags: list or list-like
        List of lists of tags to be merged. Writes 1 for all tags in the hot-encoded vector whenever at least one tag of the list is present.

    Examples
    --------
    >>> features['tags'] = [0, 1, 1, 0, 0]
    >>> _merge(features, merge_tags=[[0, 1], [2, 3]])
    features['tags']: [1, 1, 1, 1, 0]
    >>> _merge(features, merge_tags=[[0, 1], [3, 4]])
    features['tags']: [1, 1, 1, 0, 0] 
    '''
    
    merge_tags = np.array(merge_tags)

    assert len(merge_tags.shape) == 2 , 'merge_tags must be a two-dimensional array'

    n_tags = tf.cast(tf.shape(features_dict['tags']), tf.int64)

    feature_tags = tf.dtypes.cast(features_dict['tags'], tf.bool)

    for tags in merge_tags: # for each list of tags in 'merge_tags' (which is a list of lists...)
        idxs = np.subtract(np.sort(np.array(tags, dtype=np.int64)).reshape(-1, 1), 1)
        vals = np.ones(len(tags), dtype=np.int64)
        tags = tf.SparseTensor(indices=idxs, values=vals, dense_shape=n_tags)
        tags = tf.sparse.to_dense(tags)
        tags = tf.dtypes.cast(tags, tf.bool)
        # if at least one of the feature tags is in the current 'tags' list, write True in the bool-hot-encoded vector for all tags in 'tags'; otherwise, leave feature tags as they are
        features_dict['tags'] = tf.where(tf.math.reduce_any(tags & feature_tags), tags | feature_tags, feature_tags)
    features_dict['tags'] = tf.cast(features_dict['tags'], tf.float32) # cast back to float32
    return features_dict

def _tag_filter(features_dict, tags):
    ''' Removes unwanted tids from the dataset based on given tags (use with tf.data.Dataset.filter).
    
    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .filter).

    tags: list or list-like
        List containing tag idxs (as int) to be "allowed" in the output dataset.
    '''

    n_tags = tf.cast(tf.shape(features_dict['tags']), tf.int64)

    feature_tags = tf.math.equal(tf.unstack(features_dict['tags']), 1) # bool tensor where True/False correspond to has/doesn't have tag
    idxs = np.subtract(np.sort(np.array(tags, dtype=np.int64)).reshape(-1, 1), 1)
    vals = np.ones(len(tags), dtype=np.int64)
    tags_mask = tf.SparseTensor(indices=idxs, values=vals, dense_shape=n_tags)
    tags_mask = tf.sparse.to_dense(tags_mask)
    tags_mask = tf.dtypes.cast(tags_mask, tf.bool)

    return tf.math.reduce_any(feature_tags & tags_mask) # returns True if and only if at least one feature tag is in the desired 'tags' list

def _tid_filter(features_dict, tids):
    ''' Removes unwanted tids from the dataset based on given tids (use with tf.data.Dataset.filter).
        
    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .filter).

    tids: list or list-like
        List containing tids (as strings) to be "allowed" in the output dataset.
    '''

    return tf.math.reduce_any(tf.math.equal(tids, features_dict['tid']))

def _tag_filter_hotenc_mask(features_dict, tags):
    ''' Reshapes tag hot-encoded vector after filtering with _tag_filter (use with tf.data.Dataset.map).
    
    Parameters
    ----------
    features: dict
        Dict of features (as provided by .map).

    tags: list or list-like
        List containing tag idxs used for filtering with _tag_filter.
    '''

    idxs = np.subtract(np.sort(np.array(tags, dtype=np.int64)), 1)
    features_dict['tags'] = tf.gather(features_dict['tags'], idxs)
    return features_dict

def _window(features_dict, audio_format, sample_rate, window_size=15, random=False):
    ''' Extracts a window of 'window_size' seconds from the audio tensors (use with tf.data.Dataset.map).

    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .map).

    audio_format: str
        Specifies the feature audio format. Either 'waveform' or 'log-mel-spectrogram'.
    
    window_size: int
        Length (in seconds) of the desired output window.
    
    random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    '''

    if audio_format not in ('waveform', 'log-mel-spectrogram'):
        raise KeyError('invalid audio format')
    
    elif audio_format == 'waveform':
        slice_length = tf.math.multiply(tf.constant(window_size, dtype=tf.int32), tf.constant(sample_rate, dtype=tf.int32)) # get the actual slice length
        if random:
            maxval = tf.shape(features_dict['audio'], out_type=tf.int32)[0] - slice_length
            x = tf.cond(tf.equal(maxval, 0), lambda: tf.constant(0, dtype=tf.int32), lambda: tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32))
            y = x + slice_length
            features_dict['audio'] = features_dict['audio'][x:y]
        else:
            mid = tf.math.floordiv(tf.shape(features_dict['audio'], out_type=tf.int32)[0], tf.constant(2, dtype=tf.int32)) # find midpoint of audio tensors
            x = mid - tf.math.floordiv(slice_length, tf.constant(2, dtype=tf.int32))
            y = mid + tf.math.floordiv(slice_length + 1, tf.constant(2, dtype=tf.int32)) # 'slice_length + 1' ensures x:y has always length 'slice_length' regardless of whether 'slice_length' is odd or even
            features_dict['audio'] = features_dict['audio'][x:y]
    
    elif audio_format == 'log-mel-spectrogram':
        slice_length = tf.math.floordiv(tf.math.multiply(tf.constant(window_size, dtype=tf.int32), tf.constant(sample_rate, dtype=tf.int32)), tf.constant(512, dtype=tf.int32)) # get the actual slice length
        if random:
            maxval = tf.shape(features_dict['audio'], out_type=tf.int32)[1] - slice_length
            x = tf.cond(tf.equal(maxval, 0), lambda: tf.constant(0, dtype=tf.int32), lambda: tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32))
            y = x + slice_length
            features_dict['audio'] = features_dict['audio'][:,x:y]
        else:
            mid = tf.math.floordiv(tf.shape(features_dict['audio'], out_type=tf.int32)[1], tf.constant(2, dtype=tf.int32)) # find midpoint of audio tensors
            x = mid - tf.math.floordiv(slice_length, tf.constant(2, dtype=tf.int32))
            y = mid + tf.math.floordiv(slice_length + 1, tf.constant(2, dtype=tf.int32)) # 'slice_length + 1' ensures x:y has always length 'slice_length' regardless of whether 'slice_length' is odd or even
            features_dict['audio'] = features_dict['audio'][:,x:y]
    
    return features_dict

def _spect_normalization(features_dict):
    mean, variance = tf.nn.moments(features_dict['audio'], axes=[1,2], keepdims=True)
    features_dict['audio'] = tf.divide(tf.subtract(features_dict['audio'], mean), tf.sqrt(variance+0.000001))
    return features_dict

def _batch_normalization(features_dict):
    ''' Normalizes a batch to ensure zero mean and unit variance. '''

    mean, variance = tf.nn.moments(features_dict['audio'], axes=[0])
    features_dict['audio'] = tf.nn.batch_normalization(features_dict['audio'], mean, variance, offset = 0, scale = 1, variance_epsilon = .000001)
    return features_dict

def _batch_tuplification(features_dict):
    ''' Transforms a batch into (audio, tags) tuples, ready for training or evaluation with Keras. '''

    return (features_dict['audio'], features_dict['tags'])

def generate_datasets(tfrecords, audio_format, split=None, which_split=None, sample_rate=16000, batch_size=32, cycle_length=1, shuffle=True, buffer_size=10000, window_size=15, random=False, with_tids=None, with_tags=None, merge_tags=None, num_tags=155, repeat=None, as_tuple=True):
    ''' Reads the TFRecords and produces a list tf.data.Dataset objects ready for training/evaluation.
    
    Parameters:
    ----------
    tfrecords: str, list
        List of .tfrecord files paths.

    audio_format: {'waveform', 'log-mel-spectrogram'}
        Specifies the feature audio format.

    split: tuple
        Specifies the number of train/validation/test files to use when reading the .tfrecord files.
        If values add up to 100, they will be treated as percentages; otherwise, they will be treated as actual number of files to parse.

    which_split: tuple
        Applies boolean mask to the datasets obtained with split. Specifies which datasets are actually returned.

    which_split: tuple
        Applies boolean mask to the datasets obtained with split. Specifies which datasets are actually returned.

    sample_rate: int
        Specifies the sample rate used to process audio tracks.

    batch_size: int
        Specifies the dataset batch_size.

    cycle_length: int
        Controls the number of input elements that are processed concurrently.

    shuffle: bool
        If True, shuffles the dataset with buffer size = buffer_size.

    buffer_size: int
        If shuffle is True, sets the shuffle buffer size.

    window_size: int
        Specifies the desired window length (in seconds).

    random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    
    num_tags: int
        Specifies the total number of tags.

    with_tids: list
        If not None, contains the tids to be trained on.

    with_tags: list
        If not None, contains the tags to be trained on.

    merge_tags: list
        If not None, contains the lists of tags to be merged together (only applies if with_tags is specified).

    repeat: int
        If not None, repeats the dataset only for a given number of epochs (default is repeat indefinitely).

    as_tuple: bool
        If True, discards tid's and transforms features into (audio, tags) tuples.
    '''

    AUDIO_SHAPE = {'waveform': (-1, ), 'log-mel-spectrogram': (96, -1)} # set audio tensors dense shape

    AUDIO_FEATURES_DESCRIPTION = {
        'audio': tf.io.VarLenFeature(tf.float32),
        'tid': tf.io.FixedLenFeature((), tf.string),
        'tags': tf.io.FixedLenFeature((num_tags, ), tf.int64)
    }

    assert audio_format in ('waveform', 'log-mel-spectrogram') , 'invalid audio format'
    
    tfrecords = np.array(tfrecords, dtype=np.unicode) # allow for single str as input
    tfrecords = np.vectorize(lambda x: os.path.abspath(os.path.expanduser(x)))(tfrecords) # fix issues with relative paths in input list

    if split is not None:
        if np.sum(split) == 100:
            split = np.cumsum(split) * len(tfrecords) // 100
        else:
            assert np.sum(split) <= len(tfrecords) , 'split exceeds the number of available .tfrecord files'
            split = np.cumsum(split)
        tfrecords_split = np.split(tfrecords, split)
        tfrecords_split = tfrecords_split[:-1] # discard last empty split
    else:
        tfrecords_split = [tfrecords]

    datasets = []

    for files_list in tfrecords_split:
        if len(files_list) > 1: # read files in parallel (number of parallel threads specified by cycle_length)
            files = tf.data.Dataset.from_tensor_slices(files_list)
            dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=cycle_length, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = tf.data.TFRecordDataset(files_list)
            
        # parse serialized features
        dataset = dataset.map(lambda x: _parse_features(x, AUDIO_FEATURES_DESCRIPTION, AUDIO_SHAPE[audio_format]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # shuffle
        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        # apply tid and tag filters
        if with_tags is not None:
            if merge_tags is not None:
                dataset = dataset.map(lambda x: _merge(x, merge_tags))
            dataset = dataset.filter(lambda x: _tag_filter(x, with_tags)).map(lambda y: _tag_filter_hotenc_mask(y, with_tags))
        if with_tids is not None:
            dataset = dataset.filter(lambda x: _tid_filter(x, with_tids))
        
        # slice into audio windows
        dataset = dataset.map(lambda x: _window(x, audio_format, sample_rate, window_size, random), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # batch
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # normalize data
        if audio_format == 'log-mel-spectrogram':
            dataset = dataset.map(_spect_normalization, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(_batch_normalization, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # convert features from dict into tuple
        if as_tuple:
            dataset = dataset.map(_batch_tuplification, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # performance optimization

        datasets.append(dataset)
    
    if which_split is not None:
        if split is not None:
            assert len(which_split) == len(split) , 'split and which_split must have the same length'
            datasets = np.array(datasets)[np.array(which_split, dtype=np.bool)].tolist()
        else:
            datasets = datasets + [None] * (which_split.count(1) - 1) # useful when trying to unpack datasets, but split has not been provided
    
    if len(datasets) == 1:
        return datasets[0]
    else:
        return datasets

def generate_datasets_from_dir(tfrecords_dir, audio_format, split=None, which_split=None, sample_rate=16000, batch_size=32, cycle_length=1, shuffle=True, buffer_size=10000, window_size=15, random=False, with_tids=None, with_tags=None, merge_tags=None, num_tags=155, repeat=1, as_tuple=True):
    ''' Reads the TFRecords from the input directory and produces a list tf.data.Dataset objects ready for training/evaluation.
    
    Parameters:
    ----------
    tfrecords_dir: str
        Directory containing the .tfrecord files.

    split: tuple
        Specifies the number of train/validation/test files to use when reading the .tfrecord files.
        If values add up to 100, they will be treated as percentages; otherwise, they will be treated as actual number of files to parse.

    which_split: tuple
        Applies boolean mask to the datasets obtained with split. Specifies which datasets are actually returned.

    which_split: tuple
        Applies boolean mask to the datasets obtained with split. Specifies which datasets are actually returned.

    sample_rate: int
        Specifies the sample rate used to process audio tracks.

    batch_size: int
        Specifies the dataset batch_size.

    cycle_length: int
        Controls the number of input elements that are processed concurrently.

    shuffle: bool
        If True, shuffles the dataset with buffer size = buffer_size.

    buffer_size: int
        If shuffle is True, sets the shuffle buffer size.

    window_size: int
        Specifies the desired window length (in seconds).

    random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    
    num_tags: int
        Specifies the total number of tags.

    with_tids: list
        If not None, contains the tids to be trained on.

    with_tags: list
        If not None, contains the tags to be trained on.

    merge_tags: list
        If not None, contains the lists of tags to be merged together (only applies if with_tags is specified).

    repeat: int
        If not None, repeats the dataset only for a given number of epochs (default is repeat indefinitely).

    as_tuple: bool
        If True, discards tid's and transforms features into (audio, tags) tuples.
    '''

    tfrecords = []

    for file in os.listdir(os.path.expanduser(tfrecords_dir)):
        if file.endswith(".tfrecord") and file.split('_')[0] == audio_format:
            tfrecords.append(os.path.abspath(os.path.join(tfrecords_dir, file)))

    return generate_datasets(tfrecords, audio_format, split, which_split, sample_rate, batch_size, cycle_length, shuffle, buffer_size, window_size, random, with_tids, with_tags, merge_tags, num_tags, repeat, as_tuple)
