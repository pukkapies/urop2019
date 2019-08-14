''' Contains tools to read the TFRecords and generate a tf.data.Dataset.


Notes
-----
This module is meant to be imported in the training pipeline. Just run
the function generate_dataset() (see function docs below for details on the right 
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

- _tid_filter
    Remove (tracks with) unwanted tids from the dataset.

- _tad_filter
    Remove (tracks with) unwanted tags from the dataset.

- _tag_filter_hotenc_mask
    Change the shape of tag hot-encoded vector to suit the output of _tag_filter.

- _window
    Extract a sample of n seconds from each audio tensor within a batch.

- _batch_normalization
    Ensure zero mean and variation within a batch.

- _batch_tuplification
    Transform features from dict to tuple.

- _genrate_dataset
    Combine all previous functions to produce the final output dataset.
'''

import os

import numpy as np
import tensorflow as tf

N_TAGS = 155
SAMPLE_RATE = 16000

default_tfrecord_root_dir = '/srv/data/urop/tfrecords'

audio_feature_description = {
    'audio' : tf.io.VarLenFeature(tf.float32),
    'tid' : tf.io.FixedLenFeature((), tf.string),
    'tags' : tf.io.FixedLenFeature((N_TAGS, ), tf.int64)
}

def _parse_features(example):
    ''' Parses the serialized tf.Example. '''

    return tf.io.parse_single_example(example, audio_feature_description)

def _reshape(features_dict, shape):
    ''' Reshapes each flattened audio tensors into the 'correct' one. '''

    features_dict['audio'] = tf.sparse.reshape(features_dict['audio'], shape)
    return features_dict

def _tid_filter(features_dict, tids):
    ''' Removes unwanted tids from the dataset (use with tf.data.Dataset.filter).
        
    Parameters
    ----------
    features_dict : dict
        Dict of features (as provided by .filter).

    tids : list or list-like
        List containing tids (as strings) to be "allowed" in the output dataset.
    '''

    return tf.math.reduce_any(tf.math.equal(tids, features_dict['tid']))

def _tag_filter(features_dict, tags):
    ''' Removes unwanted tids from the dataset (use with tf.data.Dataset.filter).
    
    Parameters
    ----------
    features_dict : dict
        Dict of features (as provided by .filter).

    tags : list or list-like
        List containing tag idxs (as int) to be "allowed" in the output dataset.
    '''

    feature_tags = tf.math.equal(tf.unstack(features_dict['tags']), 1) # bool tensor where True/False correspond to has/doesn't have tag

    tags_mask = tf.SparseTensor(indices=np.subtract(np.array(tags, dtype=np.int64).reshape(-1, 1), 1), values=np.ones(len(tags), dtype=np.int64), dense_shape=np.array([N_TAGS], dtype=np.int64))
    tags_mask = tf.sparse.to_dense(tags_mask)
    tags_mask = tf.dtypes.cast(tags_mask, tf.bool)

    return tf.math.reduce_any(feature_tags & tags_mask) # returns True if and only if at least one feature tag is in the desired 'tags' list

def _tag_filter_hotenc_mask(features_dict, tags):
    ''' Reshapes tag hot-encoded vector after filtering with _tag_filter (use with tf.data.Dataset.map).
    
    Parameters
    ----------
    features : dict
        Dict of features (as provided by .map).

    tags : list or list-like
        List containing tag idxs used for filtering with _tag_filter.
    '''

    tags_mask = tf.SparseTensor(indices=np.subtract(np.array(tags, dtype=np.int64).reshape(-1, 1), 1), values=np.ones(len(tags), dtype=np.int64), dense_shape=np.array([N_TAGS], dtype=np.int64))
    tags_mask = tf.sparse.to_dense(tags_mask)
    tags_mask = tf.dtypes.cast(tags_mask, tf.bool)
    features_dict['tags'] = tf.boolean_mask(features_dict['tags'], tags_mask)
    return features_dict

def _window(features_dict, audio_format, window_length=15, random=False):
    ''' Extracts a window of 'window_length' seconds from the audio tensors (use with tf.data.Dataset.map).

    Parameters
    ----------
    features_dict : dict
        Dict of features (as provided by .map).

    audio_format : str
        Specifies the feature audio format. Either 'waveform' or 'log-mel-spectrogram'.
    
    window_length : int
        Length (in seconds) of the desired output window.
    
    random : bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    '''

    HOP_LENGTH = 512

    if audio_format not in ('waveform', 'log-mel-spectrogram'):
        raise KeyError()
    
    elif audio_format == 'waveform':
        features_dict['audio'] = tf.sparse.to_dense(features_dict['audio'])
        slice_length = tf.math.multiply(tf.constant(window_length, dtype=tf.int32), tf.constant(SAMPLE_RATE, dtype=tf.int32)) # get the actual slice length
        if random:
            maxval = tf.shape(features_dict['audio'], out_type=tf.int32)[1] - slice_length
            x = tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32)
            y = x + slice_length
            features_dict['audio'] = features_dict['audio'][:,x:y]
        else:
            mid = tf.math.floordiv(tf.shape(features_dict['audio'], out_type=tf.int32)[1], tf.constant(2, dtype=tf.int32)) # find midpoint of audio tensors
            x = mid - tf.math.floordiv(slice_length, tf.constant(2, dtype=tf.int32))
            y = mid + tf.math.floordiv(slice_length + 1, tf.constant(2, dtype=tf.int32)) # 'slice_length + 1' ensures x:y has always length 'slice_length' regardless of whether 'slice_length' is odd or even
            features_dict['audio'] = features_dict['audio'][:,x:y]
    
    elif audio_format == 'log-mel-spectrogram':
        features_dict['audio'] = tf.sparse.to_dense(features_dict['audio'])
        slice_length = tf.math.floordiv(tf.math.multiply(tf.constant(window_length, dtype=tf.int32), tf.constant(SAMPLE_RATE, dtype=tf.int32)), tf.constant(HOP_LENGTH, dtype=tf.int32)) # get the actual slice length
        if random:
            maxval = tf.shape(features_dict['audio'], out_type=tf.int32)[2] - slice_length
            x = tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32)
            y = x + slice_length
            features_dict['audio'] = features_dict['audio'][:,:,x:y]
        else:
            mid = tf.math.floordiv(tf.shape(features_dict['audio'], out_type=tf.int32)[2], tf.constant(2, dtype=tf.int32)) # find midpoint of audio tensors
            x = mid - tf.math.floordiv(slice_length, tf.constant(2, dtype=tf.int32))
            y = mid + tf.math.floordiv(slice_length + 1, tf.constant(2, dtype=tf.int32)) # 'slice_length + 1' ensures x:y has always length 'slice_length' regardless of whether 'slice_length' is odd or even
            features_dict['audio'] = features_dict['audio'][:,:,x:y]
    
    return features_dict

def _batch_normalization(features_dict):
    ''' Normalizes a batch to ensure zero mean and unit variance. '''

    mean, variance = tf.nn.moments(features_dict['audio'], axes=[0])
    features_dict['audio'] = tf.nn.batch_normalization(features_dict['audio'], mean, variance, offset = 0, scale = 1, variance_epsilon = .000001)
    return features_dict

def _batch_tuplification(features_dict):
    ''' Transforms a batch into (audio, tags) tuples, ready for training or evaluation with Keras. '''

    return (features_dict['audio'], features_dict['tags'])

def generate_dataset(tfrecords, audio_format, batch_size=32, shuffle=True, buffer_size=10000, window_length=15, random=False, with_tags=None, with_tids=None, num_epochs=None, as_tuple=True):
    ''' Reads the TFRecords and produce a tf.data.Dataset ready to be iterated during training/evaluation.
    
    Parameters:
    ----------
    tfrecords : str, list
        List of TFRecords to read.

    audio_format : {'waveform', 'log-mel-spectrogram'}
        Specifies the feature audio format.

    batch_size : int
        Specifies the dataset batch_size.

    shuffle : bool
        If True, shuffles the dataset with buffer size = buffer_size.

    buffer_size : int
        If shuffle is True, sets the shuffle buffer size.

    window_length : int
        Specifies the desired window length (in seconds).

    random : bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).

    with_tags : list
        If not None, contains the tags to be trained on.

    with_tids : list
        If not None, contains the tids to be trained on.

    num_epochs : int
        If not None, repeats the dataset only for a given number of epochs (default is repeat indefinitely).

    as_tuple : bool
        If True, discards tid's and transforms features into (audio, tags) tuples.
    '''

    TENSOR_SHAPE = {'waveform': (-1, ), 'log-mel-spectrogram': (96, -1)} # set audio tensors dense shape

    assert audio_format in ('waveform', 'log-mel-spectrogram')
    
    tfrecords = np.array(tfrecords, dtype=np.unicode) # allow for single str as input
    tfrecords = np.vectorize(lambda x: os.path.abspath(os.path.expanduser(x)))(tfrecords) # fix issues with relative paths in input list

    tfrecords = tf.data.Dataset.from_tensor_slices(tfrecords)
    dataset = tfrecords.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE) # load dataset, read files in parallel
    dataset = dataset.map(_parse_features, num_parallel_calls=tf.data.experimental.AUTOTUNE) # parse serialized features
    dataset = dataset.map(lambda x: _reshape(x, TENSOR_SHAPE[audio_format]), num_parallel_calls=tf.data.experimental.AUTOTUNE) # reshape
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    if with_tags:
        dataset = dataset.filter(lambda x: _tag_filter(x, with_tags)).map(lambda x: _tag_filter_hotenc_mask(x, with_tags))
    if with_tids:
        dataset = dataset.filter(lambda x: _tid_filter(x, with_tids))
    
    dataset = dataset.batch(batch_size) # create batches before slicing the desired audio window to boost performance
    dataset = dataset.map(lambda x: _window(x, audio_format, window_length, random), num_parallel_calls=tf.data.experimental.AUTOTUNE) # slice the desired audio window
    dataset = dataset.map(_batch_normalization, num_parallel_calls=tf.data.experimental.AUTOTUNE) # normalize data
    dataset = dataset.map(_batch_tuplification, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # performance optimization
    
    return dataset