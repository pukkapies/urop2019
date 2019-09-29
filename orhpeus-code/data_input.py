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

- _window_1
    Extract a sample of n seconds from each audio tensor within a batch.

- _window_2
    Extract a sample of n seconds from each audio tensor within a batch.

- _window
    Return either _window_1 if audio-format is waveform, or _window_2 if audio-format is log-mel-spectrogram.

- _spect_normalization
    Ensure zero mean and unit variance within a batch of log-mel-spectrograms.

- _batch_normalization
    Ensure zero mean and unit variance within a batch.

- _tuplify
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

    tags: list or list-like
        List of lists of tags to be merged. Writes 1 for all tags in the hot-encoded vector whenever at least one tag of the list is present.

    Examples
    --------
    >>> features['tags'] = [0, 1, 1, 0, 0]
    >>> _merge(features, merge_tags=[[0, 1], [2, 3]])
    features['tags']: [1, 1, 1, 1, 0]
    >>> _merge(features, merge_tags=[[0, 1], [3, 4]])
    features['tags']: [1, 1, 1, 0, 0] 
    '''

    tags_databases = len(features_dict) - 2 # check if multiple databases have been provided
    num_tags = tf.cast(tf.shape(features_dict['tags']), tf.int64)

    
    tags = tf.dtypes.cast(tags, tf.int64)
    idxs = tf.subtract(tf.reshape(tf.sort(tags), [-1,1]), tf.constant(1, dtype=tf.int64))
    vals = tf.constant(1, dtype=tf.int64, shape=[len(tags)])
    tags = tf.SparseTensor(indices=idxs, values=vals, dense_shape=num_tags)
    tags = tf.sparse.to_dense(tags)
    tags = tf.dtypes.cast(tags, tf.bool)
    
    def _fn(tag_str, num_tags=num_tags): # avoid repetitions of code by defining a handy function
        feature_tags = tf.dtypes.cast(features_dict[tag_str], tf.bool)
        # if at least one of the feature tags is in the current 'tags' list, write True in the bool-hot-encoded vector for all tags in 'tags'; otherwise, leave feature tags as they are
        features_dict[tag_str] = tf.where(tf.math.reduce_any(tags & feature_tags), tags | feature_tags, feature_tags)
        features_dict[tag_str] = tf.cast(features_dict[tag_str], tf.int64)

    if tags_databases > 1:
        for i in range(tags_databases):
            _fn('tags_' + str(i))
    else:
        _fn('tags')

    return features_dict

def _tag_filter(features_dict, tags, which_tags=None):
    ''' Removes unwanted tids from the dataset based on given tags (use with tf.data.Dataset.filter).
    
    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .filter).

    tags: list or list-like
        List containing tag idxs (as int) to be "allowed" in the output dataset.

    which_tags: int
        If not None, specifies the database to filter on (when multiple databases are provided).
    '''

    tags = tf.dtypes.cast(tags, dtype=tf.int64)
    
    if which_tags is None:
        dict_key = 'tags'
    else:
        assert isinstance(which_tags, int), 'which_tags must be an integer'
        dict_key = 'tags_' + str(which_tags)
    
    num_tags = tf.cast(tf.shape(features_dict[dict_key]), tf.int64)
    feature_tags = tf.math.equal(tf.unstack(features_dict[dict_key]), tf.constant(1, dtype=tf.int64)) # bool tensor where True/False correspond to has/doesn't have tag
    idxs = tf.subtract(tf.reshape(tf.sort(tags), [-1,1]), tf.constant(1, dtype=tf.int64))
    vals = tf.constant(1, dtype=tf.int64, shape=[len(tags)])
    tags_mask = tf.SparseTensor(indices=idxs, values=vals, dense_shape=num_tags)
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
    
    tags_databases = len(features_dict) - 2 # check if multiple databases have been provided
    
    tags = tf.dtypes.cast(tags, dtype=tf.int64)
    idxs = tf.subtract(tf.sort(tags), tf.constant(1, dtype=tf.int64))

    def _fn(tag_str): # avoid repetitions of code by defining a handy function
        features_dict[tag_str] = tf.gather(features_dict[tag_str], idxs)
    
    if tags_databases > 1:
        for i in range(tags_databases):
            _fn('tags_' + str(i))
    else:
        _fn('tags')

    return features_dict

def _window_1(features_dict, sample_rate, window_length=15, window_random=False):
    ''' Extracts a window of 'window_length' seconds from the audio tensors (use with tf.data.Dataset.map).

    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .map).

    sample_rate: int
        Specifies the sample rate of the audio track.
    
    window_length: int
        Length (in seconds) of the desired output window.
    
    window_random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    '''
    
    slice_length = tf.math.multiply(tf.constant(window_length, dtype=tf.int32), tf.constant(sample_rate, dtype=tf.int32)) # get the actual slice length
    slice_length = tf.reshape(slice_length, ())

    random = tf.constant(window_random, dtype=tf.bool)

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

def _window_2(features_dict, sample_rate, window_length=15, window_random=False):
    ''' Extracts a window of 'window_length' seconds from the audio tensors (use with tf.data.Dataset.map).

    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .map).

    sample_rate: int
        Specifies the sample rate of the audio track.
    
    window_length: int
        Length (in seconds) of the desired output window.
    
    window_random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    '''

    slice_length = tf.math.floordiv(tf.math.multiply(tf.constant(window_length, dtype=tf.int32), tf.constant(sample_rate, dtype=tf.int32)), tf.constant(512, dtype=tf.int32)) # get the actual slice length
    slice_length = tf.reshape(slice_length, ())

    random = tf.constant(window_random, dtype=tf.bool)

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

def _window(audio_format):
    ''' Returns the right window function, depending to the specified audio-format. '''

    return {'waveform': _window_1, 'log-mel-spectrogram': _window_2}[audio_format]

def _spect_normalization(features_dict):
    ''' Normalizes the log-mel-spectrograms within a batch. '''

    mean, variance = tf.nn.moments(features_dict['audio'], axes=[1,2], keepdims=True)
    features_dict['audio'] = tf.nn.batch_normalization(features_dict['audio'], mean, variance, offset = 0, scale = 1, variance_epsilon = .000001)
    return features_dict

def _batch_normalization(features_dict):
    ''' Normalizes a batch. '''

    mean, variance = tf.nn.moments(features_dict['audio'], axes=[0])
    features_dict['audio'] = tf.nn.batch_normalization(features_dict['audio'], mean, variance, offset = 0, scale = 1, variance_epsilon = .000001)
    return features_dict

def _tuplify(features_dict, which_tags=None):
    ''' Transforms a batch into (audio, tags) tuples, ready for training or evaluation with Keras. 
    
    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .filter).

    which_tags: int
        If not None, specifies the database to use (when multiple databases are provided).
    '''
    
    if which_tags is None:
        return (features_dict['audio'], features_dict['tags'])
    else:
        assert isinstance(which_tags, int), 'which_tags must be an integer'
        return (features_dict['audio'], features_dict['tags_' + str(which_tags)])

def generate_datasets(tfrecords, audio_format, split=None, which_split=None, sample_rate=16000, num_mels=96, batch_size=32, block_length=1, cycle_length=1, shuffle=True, shuffle_buffer_size=10000, window_length=15, window_random=False, with_tids=None, with_tags=None, merge_tags=None, num_tags=155, num_tags_db=1, default_tags_db=None, repeat=None, as_tuple=True):
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

    sample_rate: int
        Specifies the sample rate used to process audio tracks.

    batch_size: int
        Specifies the dataset batch_size.

    block_length: int
        Controls the number of input elements that are processed concurrently.

    cycle_length: int
        Controls the number of input elements that are processed concurrently.

    shuffle: bool
        If True, shuffles the dataset with buffer size = shuffle_buffer_size.

    shuffle_buffer_size: int
        If shuffle is True, sets the shuffle buffer size.

    window_length: int
        Specifies the desired window length (in seconds).

    window_random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).

    num_mels: int
        The number of mels in the mel-spectrogram.
    
    num_tags: int
        The total number of tags.
    
    num_tags_db: int
        The total number of tags databases used.
    
    default_tags_db: int
        Specifies the tags database to use when filtering tags or converting into tuple (if multiple databases are provided).

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

    AUDIO_SHAPE = {'waveform': (-1, ), 'log-mel-spectrogram': (num_mels, -1)} # set audio tensors dense shape

    AUDIO_FEATURES_DESCRIPTION = {'audio': tf.io.VarLenFeature(tf.float32), 'tid': tf.io.FixedLenFeature((), tf.string)} # tags will be added just below

    # check if multiple databases have been provided
    if num_tags_db == 1:
        
        # add feature 'tags'
        AUDIO_FEATURES_DESCRIPTION['tags'] = tf.io.FixedLenFeature((num_tags, ), tf.int64)
    
    else:
        default_tags_db = default_tags_db or 0
        
        # add feature 'tags_i' for each i-th tags database provided
        for i in range(num_tags_db):
            AUDIO_FEATURES_DESCRIPTION['tags_' + str(i)] = tf.io.FixedLenFeature((num_tags, ), tf.int64)

    assert audio_format in ('waveform', 'log-mel-spectrogram'), 'please provide a valid audio format'
    
    tfrecords = np.array(tfrecords, dtype=np.unicode) # allow for single str as input
    tfrecords = np.vectorize(lambda x: os.path.abspath(os.path.expanduser(x)))(tfrecords) # fix issues with relative paths in input list

    if split:
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
        if files_list.size > 1: # read files in parallel (number of parallel threads specified by cycle_length)
            files = tf.data.Dataset.from_tensor_slices(files_list)
            dataset = files.interleave(tf.data.TFRecordDataset, block_length=block_length, cycle_length=cycle_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = tf.data.TFRecordDataset(files_list)
        
        # parse serialized features
        dataset = dataset.map(lambda x: _parse_features(x, AUDIO_FEATURES_DESCRIPTION, AUDIO_SHAPE[audio_format]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                
        # shuffle
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)

        # apply filters
        if with_tags is not None:
            if merge_tags is not None:
                for tags in merge_tags:
                      dataset = dataset.map(lambda x: _merge(x, tags))
            dataset = dataset.filter(lambda x: _tag_filter(x, tags=with_tags, which_tags=default_tags_db)).map(lambda y: _tag_filter_hotenc_mask(y, tags=with_tags))
                        
        if with_tids is not None:
            dataset = dataset.filter(lambda x: _tid_filter(x, tids=with_tids))
        
        # slice into audio windows
        dataset = dataset.map(lambda x: _window(audio_format)(x, sample_rate, window_length, window_random), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # batch
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # normalize data
        if audio_format == 'log-mel-spectrogram':
            dataset = dataset.map(_spect_normalization, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(_batch_normalization, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # convert features from dict into tuple
        if as_tuple:
            dataset = dataset.map(lambda x: _tuplify(x, which_tags=default_tags_db), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # performance optimization

        datasets.append(dataset)
    
    if which_split is not None:
        if split is not None:
            assert len(which_split) == len(split), 'split and which_split must have the same length'
            datasets = np.array(datasets)[np.array(which_split, dtype=np.bool)].tolist()
        else:
            datasets = datasets + [None] * (which_split.count(1) - 1) # useful when trying to unpack datasets, but split has not been provided
    
    if len(datasets) == 1:
        return datasets[0]
    else:
        return datasets

def generate_datasets_from_dir(tfrecords_dir, audio_format, split=None, which_split=None, sample_rate=16000, num_mels=96, batch_size=32, block_length=1, cycle_length=1, shuffle=True, shuffle_buffer_size=10000, window_length=15, window_random=False, with_tids=None, with_tags=None, merge_tags=None, num_tags=155, num_tags_db=1, default_tags_db=None, repeat=1, as_tuple=True):
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

    sample_rate: int
        Specifies the sample rate used to process audio tracks.

    batch_size: int
        Specifies the dataset batch_size.

    block_length: int
        Controls the number of input elements that are processed concurrently.

    cycle_length: int
        Controls the number of input elements that are processed concurrently.

    shuffle: bool
        If True, shuffles the dataset with buffer size = shuffle_buffer_size.

    shuffle_buffer_size: int
        If shuffle is True, sets the shuffle buffer size.

    window_length: int
        Specifies the desired window length (in seconds).

    window_random: bool
        Specifies how the window is to be extracted. If True, slices the window randomly (default is pick from the middle).
    
    num_mels: int
        The number of mels in the mel-spectrogram.
    
    num_tags: int
        The total number of tags.
    
    num_tags_db: int
        The total number of tags databases used.
    
    default_tags_db: int
        Specifies the tags database to use when filtering tags or converting into tuple (if multiple databases are provided).

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

    return generate_datasets(tfrecords, audio_format, split=split, which_split=which_split, 
                             sample_rate = sample_rate, batch_size = batch_size, 
                             block_length = block_length, cycle_length = cycle_length, shuffle = shuffle, shuffle_buffer_size = shuffle_buffer_size, 
                             window_length = window_length, window_random = window_random, 
                             num_mels = num_mels, num_tags = num_tags, with_tids = with_tids, with_tags = with_tags, merge_tags = merge_tags, num_tags_db = num_tags_db, default_tags_db = default_tags_db,
                             repeat = repeat, as_tuple = as_tuple)
