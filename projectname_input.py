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

- _window_waveform
    Extract a sample of n seconds from each audio tensor within a batch (use with waveform).

- _window_log_mel_spectrogram
    Extract a sample of n seconds from each audio tensor within a batch (use with log_mel_spectrogram).

- _spect_normalization
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

def _parse_features(example, features_dict, shape):
    ''' Parses the serialized tf.Example. '''

    features_dict = tf.io.parse_single_example(example, features_dict)
    features_dict['audio'] = tf.reshape(tf.sparse.to_dense(features_dict['audio']), shape)
    return features_dict

def _merge(features_dict, tags):
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
    tags = tf.dtypes.cast(tags, tf.int64)
    n_tags = tf.cast(tf.shape(features_dict['tags']), tf.int64)

    feature_tags = tf.dtypes.cast(features_dict['tags'], tf.bool)
    
    idxs = tf.subtract(tf.reshape(tf.sort(tags), [-1,1]), tf.constant(1, dtype=tf.int64))
    vals = tf.constant(1, dtype=tf.int64, shape=[len(tags)])
    tags = tf.SparseTensor(indices=idxs, values=vals, dense_shape=n_tags)
    tags = tf.sparse.to_dense(tags)
    tags = tf.dtypes.cast(tags, tf.bool)
    # if at least one of the feature tags is in the current 'tags' list, write True in the bool-hot-encoded vector for all tags in 'tags'; otherwise, leave feature tags as they are
    features_dict['tags'] = tf.where(tf.math.reduce_any(tags & feature_tags), tags | feature_tags, feature_tags)
    features_dict['tags'] = tf.cast(features_dict['tags'], tf.int64)
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
    tags = tf.dtypes.cast(tags, dtype=tf.int64)
    n_tags = tf.cast(tf.shape(features_dict['tags']), tf.int64)

    feature_tags = tf.math.equal(tf.unstack(features_dict['tags']), tf.constant(1, dtype=tf.int64)) # bool tensor where True/False correspond to has/doesn't have tag
    idxs = tf.subtract(tf.reshape(tf.sort(tags), [-1,1]), tf.constant(1, dtype=tf.int64))
    vals = tf.constant(1, dtype=tf.int64, shape=[len(tags)])
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
    tids = tf.constant(tids, tf.string)
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
    tags = tf.dtypes.cast(tags, dtype=tf.int64)
    idxs = tf.subtract(tf.sort(tags), tf.constant(1, dtype=tf.int64))
    features_dict['tags'] = tf.gather(features_dict['tags'], idxs)
    return features_dict

def _window_waveform(features_dict, sample_rate, window_size=15, random=False):
    ''' Extracts a window of 'window_size' seconds from the audio tensors (use with tf.data.Dataset.map).

    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .map).

    sample_rate: int
        Specifies the sample rate of the audio track.
    
    window_size: int
        Length (in seconds) of the desired output window.
    
    random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    '''
    
    slice_length = tf.math.multiply(tf.constant(window_size, dtype=tf.int32), tf.constant(sample_rate, dtype=tf.int32)) # get the actual slice length
    slice_length = tf.reshape(slice_length, ())

    random = tf.constant(random, dtype=tf.bool)

    def fn1a(audio, slice_length=slice_length):
        maxval = tf.subtract(tf.shape(audio, out_type=tf.int32)[0], slice_length)
        x = tf.cond(tf.equal(maxval, tf.constant(0)), lambda: tf.constant(0, dtype=tf.int32), lambda: tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32))
        y = tf.add(x, slice_length)
        audio = audio[x:y]
        return audio
        
    def fn1b(audio):
        mid = tf.math.floordiv(tf.shape(audio, out_type=tf.int32)[0], tf.constant(2, dtype=tf.int32)) # find midpoint of audio tensors
        x = tf.subtract(mid, tf.math.floordiv(slice_length, tf.constant(2, dtype=tf.int32)))
        y = tf.add(mid, tf.math.floordiv(tf.add(slice_length, tf.constant(1)), tf.constant(2, dtype=tf.int32))) # 'slice_length + 1' ensures x:y has always length 'slice_length' regardless of whether 'slice_length' is odd or even
        audio = audio[x:y]
        return audio

    features_dict['audio'] = tf.cond(random, lambda: fn1a(features_dict['audio']), lambda: fn1b(features_dict['audio']))
    return features_dict

def _window_log_mel_spectrogram(features_dict, sample_rate, window_size=15, random=False):
    ''' Extracts a window of 'window_size' seconds from the audio tensors (use with tf.data.Dataset.map).

    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .map).

    sample_rate: int
        Specifies the sample rate of the audio track.
    
    window_size: int
        Length (in seconds) of the desired output window.
    
    random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    '''

    slice_length = tf.math.floordiv(tf.math.multiply(tf.constant(window_size, dtype=tf.int32), tf.constant(sample_rate, dtype=tf.int32)), tf.constant(512, dtype=tf.int32)) # get the actual slice length
    slice_length = tf.reshape(slice_length, ())

    random = tf.constant(random, dtype=tf.bool)

    def fn2a(audio, slice_length=slice_length):
        maxval = tf.subtract(tf.shape(audio, out_type=tf.int32)[1], slice_length)
        x = tf.cond(tf.equal(maxval, tf.constant(0)), lambda: tf.constant(0, dtype=tf.int32), lambda: tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32))
        x = tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32)
        y = tf.add(x, slice_length)
        audio = audio[:,x:y]
        return audio
        
    def fn2b(audio):
        mid = tf.math.floordiv(tf.shape(audio, out_type=tf.int32)[1], tf.constant(2, dtype=tf.int32)) # find midpoint of audio tensors
        x = tf.subtract(mid, tf.math.floordiv(slice_length, tf.constant(2, dtype=tf.int32)))
        y = tf.add(mid, tf.math.floordiv(tf.add(slice_length, tf.constant(1)), tf.constant(2, dtype=tf.int32))) # 'slice_length + 1' ensures x:y has always length 'slice_length' regardless of whether 'slice_length' is odd or even
        audio = audio[:,x:y]
        return audio
        
    features_dict['audio'] = tf.cond(random, lambda: fn2a(features_dict['audio']), lambda: fn2b(features_dict['audio']))
    return features_dict

def _spect_normalization(features_dict):
    mean, variance = tf.nn.moments(features_dict['audio'], axes=[1,2], keepdims=True)
    features_dict['audio'] = tf.divide(tf.subtract(features_dict['audio'], mean), tf.sqrt(variance+tf.constant(0.000001)))
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
        Specifies the number of train/validation/test files to use when reading the .tfrecord files (can be a tuple of any length, as long as enough files are provided in the 'tfrecords' list).

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
        assert tfrecords.size >= sum(split) , 'too few .tfrecord files to apply split'
        split = np.cumsum(split)
        tfrecords_split = np.split(tfrecords, split)
        tfrecords_split = tfrecords_split[:-1] # discard last 'empty' split
    else:
        tfrecords_split = [tfrecords]

    datasets = []

    for files_list in tfrecords_split:
        if files_list.size > 1: # read files in parallel (number of parallel threads specified by cycle_length)
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
                for tags in merge_tags:
                      dataset = dataset.map(lambda x: _merge(x, tags))
                    
            dataset = dataset.filter(lambda x: _tag_filter(x, with_tags)).map(lambda y: _tag_filter_hotenc_mask(y, with_tags))
                        
        if with_tids is not None:
            dataset = dataset.filter(lambda x: _tid_filter(x, with_tids))
        
        # slice into audio windows
        
        if audio_format == 'waveform':
            dataset = dataset.map(lambda x : _window_waveform(x, sample_rate, window_size, random), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        elif audio_format == 'log-mel-spectrogram':
            dataset = dataset.map(lambda x : _window_log_mel_spectrogram(x, sample_rate, window_size, random), num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
        Specifies the number of train/validation/test files to use when reading the .tfrecord files (can be a tuple of any length, as long as enough files are provided in the 'tfrecords' list).

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

    return generate_datasets(tfrecords, audio_format, split=split, whici_split=which_split, 
                             sample_rate=sample_rate, batch_size=batch_size, cycle_length=cycle_length, 
                             shuffle=shuffle, buffer_size=buffer_size, 
                             window_size=window_size, random=random, 
                             with_tids=with_tids, with_tags=with_tags, merge_tags=merge_tags, num_tags=num_tags, 
                             repeat=repeat, as_tuple=as_tuple)