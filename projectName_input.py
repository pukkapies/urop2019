import os

import numpy as np
import tensorflow as tf

tfrecord_root_dir = '/srv/data/urop/7digital-tfrecords'

def set_tfrecords_root_dir(new_root_dir): 
    ''' Function to set new tfrecord_root_dir. '''

    global tfrecord_root_dir
    tfrecord_root_dir = new_root_dir

audio_feature_description = {
    'audio' : tf.io.VarLenFeature(tf.float32),
    'tid' : tf.io.FixedLenFeature((), tf.string),
    'tags' : tf.io.FixedLenFeature((155, ), tf.int64) # 155 is the number of tags in the clean database
}

def _parse_audio(example):
    return tf.io.parse_single_example(example, audio_feature_description)

def _tid_filter(features, tids):
    return tf.reduce_any(tf.equal(tids, features['tid']))

def _tag_filter(features, tags):
    tags = tf.equal(tf.unstack(features['tags']), 1)
    tags_mask = tf.SparseTensor(indices=np.array(tags, dtype=np.int64).reshape(-1, 1), values=np.ones(len(tags), dtype=np.int64), dense_shape=np.array([155], dtype=np.int64))
    tags_mask = tf.sparse.to_dense(tags_mask)
    tags_mask = tf.dtypes.cast(tags_mask, tf.bool)
    return tf.reduce_any(tags & tags_mask)

def _tag_filter_hotenc_mask(features, tags):
    tags_mask = tf.SparseTensor(indices=np.array(tags, dtype=np.int64).reshape(-1, 1), values=np.ones(len(tags), dtype=np.int64), dense_shape=np.array([155], dtype=np.int64))
    tags_mask = tf.sparse.to_dense(tags_mask)
    tags_mask = tf.dtypes.cast(tags_mask, tf.bool)
    features['tags'] = tf.boolean_mask(features['tags'], tags_mask)
    return features

def _shape(features, shape = 96):
    if isinstance(shape, int):
        shape = (shape, -1)

    features['audio'] = tf.sparse.reshape(features['audio'], shape)
    return features

def _slice(features, audio_format, window_size=15, where='middle'):
    
    sr = 16000 # sample rate
    hop_length = 512 # hop length when creating mel_spectrogram
    
    if audio_format == 'waveform':
        length = features['audio'].shape[0]
        slice_length = sr*window_size

        if where == 'middle':
            features['audio'] = tf.sparse.to_dense(features['audio'][length-slice_length//2:length+slice_length//2])

        elif where == 'beginning':
            features['audio'] = tf.sparse.to_dense(features['audio'][:slice_length])

        elif where == 'end':
            features['audio'] = tf.sparse.to_dense(features['audio'][-slice_length:])

        elif where == 'random':
            s = np.random.randint(0, length-slice_length)
            features['audio'] = tf.sparse.to_dense(features['audio'][s:s+slice_length])

        else:
            print("Please enter a valid location!")
            exit()

    elif audio_format == 'log-mel-spectrogram':
        slice_length = sr*window_size//hop_length 
        length = features['audio'].shape[1]

        if where == 'middle':
            features['audio'] = tf.sparse.to_dense(features['audio'])[:,length-slice_length//2:length+slice_length//2]

        elif where == 'beginning':
            features['audio'] = tf.sparse.to_dense(features['audio'])[:,:slice_length]

        elif where == 'end':
                'audio': tf.sparse.to_dense(features['audio'])[:,-slice_length:]

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

def genrate_dataset(root_dir=tfrecord_root_dir, audio_format, window_location='middle', shuffle=True, batch_size=32, buffer_size=10000, window_size=15, reshape=None, with_tags=None, with_tids=None, num_epochs=None):
    if root_dir:
        set_tfrecords_root_dir(os.path.abspath(os.path.expanduser(root_dir)))

    tfrecords = []

    for file in os.listdir(tfrecord_root_dir):
        if file.endswith(".tfrecord") and file.split('_')[0] == audio_format:
            tfrecords.append(os.path.abspath(os.path.join(tfrecord_root_dir, file)))

    dataset = tf.data.TFRecordDataset(tfrecords).map(_parse_audio)
    
    if with_tags:
        dataset = dataset.filter(lambda x: _tag_filter(x, with_tags))
    if with_tids:
        dataset = dataset.filter(lambda x: _tid_filter(x, with_tids))
    if reshape:
        dataset = dataset.map(lambda x: _shape(x, reshape))
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    return dataset.map(lambda x: _window(x, audio_format, window_size, window_location)).batch(batch_size).map(_batch_normalization).repeat(num_epochs)
