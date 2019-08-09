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
- _parse_audio
    Parse the serialized tf.Example.

- _tid_filter
    Remove (tracks with) unwanted tids from the dataset.

- _tad_filter
    Remove (tracks with) unwanted tags from the dataset.

- _tag_filter_hotenc_mask
    Change the shape of tag hot-encoded vector to suit the output of _tag_filter.

- _shape
    Reshape parsed tf.Example if necessary (mel-spectograms were previously flattened...).

- _slice
    Extract a sample of n seconds (default is 15) from each track.

- _genrate_dataset
    Combine all previous functions to produce the final output.
'''

import os

import numpy as np
import tensorflow as tf

MEL_SPEC_HOP_LENGTH = 512
MEL_SPEC_WINDOW_SIZE = 96
N_TAGS = 155
SAMPLE_RATE = 16000

default_tfrecord_root_dir = '/srv/data/urop/tfrecords'

audio_feature_description = {
    'audio' : tf.io.VarLenFeature(tf.float32),
    'tid' : tf.io.FixedLenFeature((), tf.string),
    'tags' : tf.io.FixedLenFeature((N_TAGS, ), tf.int64)
}

def _parse_audio(example):
    return tf.io.parse_single_example(example, audio_feature_description)

def _tid_filter(features, tids):
    ''' Removes unwanted tids from the dataset (use with tf.data.Dataset.filter).
        
    Parameters
    ----------
    features : dict
        Dict of features (as provided by .filter).

    tids : list or list-like
        List containing tids (as strings) to be "allowed" in the output dataset.
    '''

    return tf.reduce_any(tf.equal(tids, features['tid']))

def _tag_filter(features, tags):
    ''' Removes unwanted tids from the dataset (use with tf.data.Dataset.filter).
    
    Parameters
    ----------
    features : dict
        Dict of features (as provided by .filter).

    tags : list or list-like
        List containing tag idxs (as int) to be "allowed" in the output dataset.
    '''

    tags_bool = tf.equal(tf.unstack(features['tags']), 1) # bool tensor where True/False correspond to has/doesn't have tag
    tags_mask = tf.SparseTensor(indices=np.array(tags, dtype=np.int64).reshape(-1, 1), values=np.ones(len(tags), dtype=np.int64), dense_shape=np.array([N_TAGS], dtype=np.int64))
    tags_mask = tf.sparse.to_dense(tags_mask)
    tags_mask = tf.dtypes.cast(tags_mask, tf.bool)
    return tf.reduce_any(tags_bool & tags_mask) # returns True if and only if at least one feature tag is in the desired 'tags' list

def _tag_filter_hotenc_mask(features, tags):
    ''' Reshapes tag hot-encoded vector after filtering with _tag_filter (use with tf.data.Dataset.map).
    
    Parameters
    ----------
    features : dict
        Dict of features (as provided by .map).

    tags : list or list-like
        List containing tag idxs used for filtering with _tag_filter.
    '''

    tags_mask = tf.SparseTensor(indices=np.array(tags, dtype=np.int64).reshape(-1, 1), values=np.ones(len(tags), dtype=np.int64), dense_shape=np.array([N_TAGS], dtype=np.int64))
    tags_mask = tf.sparse.to_dense(tags_mask)
    tags_mask = tf.dtypes.cast(tags_mask, tf.bool)
    features['tags'] = tf.boolean_mask(features['tags'], tags_mask)
    return features

def _shape(features, shape = MEL_SPEC_WINDOW_SIZE):
    ''' Reshapes the audio tensor into (shape, -1) (use with tf.data.Dataset.map). '''

    if isinstance(shape, int):
        shape = (shape, -1)

    features['audio'] = tf.sparse.reshape(features['audio'], shape)
    return features

def _slice(features, audio_format, window_length=15, random=False):
    ''' Extracts a window of 'window_length' seconds from the audio tensor (use with tf.data.Dataset.map).

    Parameters
    ----------
    features : dict
        Dict of features (as provided by .map).

    audio_format : str
        Specifies the feature audio format. Either 'waveform' or 'log-mel-spectrogram'.
    
    window_length : int
        Length (in seconds) of the desired output window.
    
    random : bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    '''

    features['audio'] = tf.sparse.to_dense(features['audio']) # convert the sparse tensor to dense tensor
    
    if audio_format not in ('waveform', 'log-mel-spectrogram'):
        raise KeyError()
    
    elif audio_format == 'waveform':
        slice_length = tf.math.multiply(tf.constant(window_length, dtype=tf.int32), tf.constant(SAMPLE_RATE, dtype=tf.int32)) # get the actual slice length
        if random:
            maxval = tf.shape(features['audio'], out_type=tf.int32)[0] - slice_length
            x = tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32)
            y = x + slice_length
            features['audio'] = features['audio'][x:y]
        else:
            mid = tf.math.floordiv(tf.shape(features['audio'], out_type=tf.int32)[0], tf.constant(2, dtype=tf.int32)) # find midpoint of audio tensor
            x = mid - tf.math.floordiv(slice_length, tf.constant(2, dtype=tf.int32))
            y = mid + tf.math.floordiv(slice_length + 1, tf.constant(2, dtype=tf.int32)) # 'slice_length + 1' ensures x:y has always length 'slice_length' regardless of whether 'slice_length' is odd or even
            features['audio'] = features['audio'][x:y]
    
    elif audio_format == 'log-mel-spectrogram':
        slice_length = tf.math.floordiv(tf.math.multiply(tf.constant(window_length, dtype=tf.int32), tf.constant(SAMPLE_RATE, dtype=tf.int32)), tf.constant(MEL_SPEC_HOP_LENGTH, dtype=tf.int32))
        if random:
            maxval = tf.shape(features['audio'], out_type=tf.int32)[1] - slice_length
            x = tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32)
            y = x + slice_length
            features['audio'] = features['audio'][:,x:y]
        else:
            mid = tf.math.floordiv(tf.shape(features['audio'], out_type=tf.int32)[1], tf.constant(2, dtype=tf.int32)) # find midpoint of audio tensor
            x = mid - tf.math.floordiv(slice_length, tf.constant(2, dtype=tf.int32))
            y = mid + tf.math.floordiv(slice_length + 1, tf.constant(2, dtype=tf.int32)) # 'slice_length + 1' ensures x:y has always length 'slice_length' regardless of whether 'slice_length' is odd or even
            features['audio'] = features['audio'][:,x:y]
    
    return features

def genrate_dataset(audio_format, root_dir=default_tfrecord_root_dir, any=False, batch_size=32, shuffle=True, buffer_size=10000, window_length=15, random=False, reshape=96, with_tags=None, with_tids=None, num_epochs=None):
    ''' Reads the TFRecords and produce a tf.data.Dataset ready to be iterated during training/evaluation.
    
    Parameters:
    ----------
    audio_format : {'waveform', 'log-mel-spectrogram'}
        Specifies the feature audio format.

    root_dir : str
        Specifies the path to the directory containing the TFRecords.

    any : bool
        Overrides default convention and import all TFRecords in the specified folder.

    batch_size : int
        Specifies the dataset batch_size.

    shuffle : bool
        If True, shuffles the dataset with buffer size = buffer_size.

    buffer_size : int
        If shuffle = True, set shuffle buffer size.

    window_length : int
        Sets the desired window length (in seconds).

    random : bool
    Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).

    reshape : int
        If audio_format is not 'waveform', specifies the shape to reshape the feature audio into.

    with_tags : list
        If not None, contains the tags to be trained on.

    with_tids : list
        If not None, contains the tids to be trained on.

    num_epochs : int
        If not None, repeats the dataset only for a given number of epochs (default is repeat indefinitely).
    '''

    assert audio_format in ('waveform', 'log-mel-spectrogram')
    
    tfrecords_root_dir = os.path.abspath(os.path.expanduser(root_dir))

    if not any:
        files = tf.data.Dataset.list_files(os.path.join(tfrecords_root_dir + '-' + audio_format, audio_format + '_*.tfrecord')) # follows convention used on our server
    else:
        files = tf.data.Dataset.list_files(os.path.join(tfrecords_root_dir, '*.tfrecord'))

    dataset = files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(_parse_audio, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size)
    
    if with_tags:
        dataset = dataset.filter(lambda x: _tag_filter(x, with_tags)).map(lambda x: _tag_filter_hotenc_mask(x, with_tags))
    if with_tids:
        dataset = dataset.filter(lambda x: _tid_filter(x, with_tids))
    
    if audio_format != 'waveform':
        dataset = dataset.map(lambda x: _shape(x, reshape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(lambda x: _slice(x, audio_format, window_length, random), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset
