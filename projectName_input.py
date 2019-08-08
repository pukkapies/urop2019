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
    Extract a sample of n seconds (default 15) from each track.

- _genrate_dataset
    Combine all previous functions to produce the final output.
'''

import os

import numpy as np
import tensorflow as tf

default_tfrecord_root_dir = '/srv/data/urop/tfrecords'

audio_feature_description = {
    'audio' : tf.io.VarLenFeature(tf.float32),
    'tid' : tf.io.FixedLenFeature((), tf.string),
    'tags' : tf.io.FixedLenFeature((155, ), tf.int64) # 155 is the number of tags in the clean database
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

    tag_bool = tf.equal(tf.unstack(features['tags']), 1)
    tags_mask = tf.SparseTensor(indices=np.array(tags, dtype=np.int64).reshape(-1, 1), values=np.ones(len(tags), dtype=np.int64), dense_shape=np.array([155], dtype=np.int64))
    tags_mask = tf.sparse.to_dense(tags_mask)
    tags_mask = tf.dtypes.cast(tags_mask, tf.bool)
    return tf.reduce_any(tag_bool & tags_mask)

def _tag_filter_hotenc_mask(features, tags):
    ''' Reshapes tag hot-encoded vector after filtering with _tag_filter (use with tf.data.Dataset.map).
    
    Parameters
    ----------
    features : dict
        Dict of features (as provided by .map).

    tags : list or list-like
        List containing tag idxs used for filtering with _tag_filter.
    '''

    tags_mask = tf.SparseTensor(indices=np.array(tags, dtype=np.int64).reshape(-1, 1), values=np.ones(len(tags), dtype=np.int64), dense_shape=np.array([155], dtype=np.int64))
    tags_mask = tf.sparse.to_dense(tags_mask)
    tags_mask = tf.dtypes.cast(tags_mask, tf.bool)
    features['tags'] = tf.boolean_mask(features['tags'], tags_mask)
    return features

def _shape(features, shape = 96):
    ''' Reshapes the sparse tensor features['audio'] into (shape, -1) (use with tf.data.Dataset.map). '''

    if isinstance(shape, int):
        shape = (shape, -1)

    features['audio'] = tf.sparse.reshape(features['audio'], shape)
    return features

def _slice(features, audio_format, window_size=15, where='middle'):
    ''' Extracts a window of the window_size seconds from the sparse tensor features['audio'] (use with tf.data.Dataset.map).

    Parameters
    ----------
    features : dict
        Dict of features (as provided by .map).

    audio_format : str
        Specifies the feature audio format. Either 'waveform' or 'log-mel-spectrogram'.
    
    window_size : int
        Length (in seconds) of the desired output window.
    
    where : str
        Specifies how the window is to be extracted. Either 'middle', or 'beginning', or 'end', or 'random'.
    '''

    SR = 16000 # sample rate
    HOP_LENGTH = 512 # hop length when creating log-mel-spectrogram
    
    if audio_format == 'waveform':
        length = features['audio'].shape[0]
        slice_length = window_size*SR

        if where == 'middle':
            features['audio'] = tf.sparse.to_dense(features['audio'])[length-slice_length//2:length+slice_length//2]

        elif where == 'beginning':
            features['audio'] = tf.sparse.to_dense(features['audio'])[:slice_length]

        elif where == 'end':
            features['audio'] = tf.sparse.to_dense(features['audio'])[-slice_length:]

        elif where == 'random':
            s = np.random.randint(0, length-slice_length)
            features['audio'] = tf.sparse.to_dense(features['audio'])[s:s+slice_length]

        else:
            print("Please enter a valid location!")
            exit()
            
    elif audio_format == 'log-mel-spectrogram':
        length = features['audio'].shape[1]
        slice_length = window_size*SR//HOP_LENGTH 

        if where == 'middle':
            features['audio'] = tf.sparse.to_dense(features['audio'])[:,length-slice_length//2:length+slice_length//2]

        elif where == 'beginning':
            features['audio'] = tf.sparse.to_dense(features['audio'])[:,:slice_length]

        elif where == 'end':
            features['audio'] = tf.sparse.to_dense(features['audio'])[:,-slice_length:]

        elif where == 'random':
            s = np.random.randint(0, length-slice_length)
            features['audio'] = tf.sparse.to_dense(features['audio'])[:,s:s+slice_length]

        else:
            print("Please enter a valid location!")
            exit()
    else:
        print("Please enter a valid audio format!")
        exit()
    
    return features

def genrate_dataset(audio_format, root_dir=default_tfrecord_root_dir, batch_size=32, shuffle=True, buffer_size=10000, window_size=15, window_location='middle', reshape=None, with_tags=None, with_tids=None, num_epochs=None):
    ''' Reads the TFRecords and produce a tf.data.Dataset ready to be iterated during training/evaluation.
    
    Parameters:
    ----------
    audio_format : {'waveform', 'log-mel-spectrogram'}
        Specifies the feature audio format.

    root_dir : str
        Specifies the path to the directory containing the TFRecords.

    batch_size : int
        Specifies the dataset batch_size.

    shuffle : bool
        If True, shuffles the dataset with buffer size = buffer_size.

    buffer_size : int
        If shuffle = True, set shuffle buffer size.

    window_size : int
        Sets the desired window length (in seconds).

    window_location : {'middle', 'beginning', 'end', 'random'}
        Specifies how the window is to be extracted.

    reshape : int
        If not None, specifies the shape to reshape the feature audio into.

    with_tags : list
        If not None, contains the tags to be trained on.

    with_tids : list
        If not None, contains the tids to be trained on.

    num_epochs : int
        If not None, repeats the dataset only for a given number of epochs (by default repeats indefinitely).
    '''

    if root_dir:
        tfrecords_root_dir = os.path.normpath(os.path.expanduser(root_dir))
    else:
        tfrecords_root_dir = os.path.normpath(default_tfrecord_root_dir + '-' + audio_format) # follows the folder structure used on our server (specify root_dir explicitely otherwise)

    files = tf.data.Dataset.list_files(os.path.join(tfrecords_root_dir, audio_format + '_*.tfrecord'))
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_reads, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(_parse_audio, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if with_tags:
        dataset = dataset.filter(lambda x: _tag_filter(x, with_tags)).map(lambda x: _tag_filter_hotenc_mask(x, with_tags))
    if with_tids:
        dataset = dataset.filter(lambda x: _tid_filter(x, with_tids))
    if reshape:
        dataset = dataset.map(lambda x: _shape(x, reshape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    return dataset.map(lambda x: _slice(x, audio_format, window_size, window_location), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).repeat(num_epochs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
