import os

import numpy as np
import tensorflow as tf

tfrecord_root_dir = '/srv/data/urop/tf'

def set_tfrecords_root_dir(new_root_dir): 
    ''' Function to set new tfrecord_root_dir. '''

    global tfrecord_root_dir
    tfrecord_root_dir = new_root_dir

audio_feature_description = {
    'audio': tf.io.VarLenFeature(tf.float32),
    'tid': tf.io.VarLenFeature(tf.string),
    'tag': tf.io.VarLenFeature(tf.int64)
}

def _parse_audio(example):
    return tf.io.parse_single_example(example, audio_feature_description)

def _reshape(data, new_shape):
    if isinstance(new_shape, int):
        new_shape = (new_shape, -1)

    return {
        'audio': tf.sparse.reshape(data['audio'], new_shape),
        'tid': data['tid'],
        'tag': data['tag']}

def _tag_filter(data, idxs):
    tags = tf.unstack(data['tag'])
    tags = tags.indices.numpy().flatten()
    return np.any(tags == idxs)

def _tid_filter(data, tids):
    tid = tf.unstack(data['tid'])
    return np.any(tids == tid)

def _window(data, s):
    pass

def genrate_dataset(mode = 'train', root_dir = tfrecord_root_dir, shuffle = True, batch_size = 32, buffer_size = 1024, window_size = 15, reshape = None, with_tags = None, with_tids = None, num_epochs = None):
    assert mode in ('train', 'valid')

    if root_dir:
        set_tfrecords_root_dir(os.path.abspath(os.path.expanduser(root_dir)))

    tfrecords = []

    for file in os.listdir(tfrecord_root_dir):
        if file.endswith(".tfrecord"):
            tfrecords.append(os.path.abspath(os.path.join(tfrecord_root_dir, file)))

    dataset = tf.data.TFRecordsDataset(tfrecords).map(_parse_audio)
    
    if with_tags:
        dataset = dataset.filter(lambda x: _tag_filter(x, with_tags))
    if with_tids:
        dataset = dataset.filter(lambda x: _tid_filter(x, with_tids))
    if reshape:
        dataset = dataset.map(lambda x: _reshape(x, reshape))
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    return dataset.map(lambda x: _window(x, window_size)).batch(batch_size).repeat(num_epochs)