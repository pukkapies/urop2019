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

def _slice(features, where='middle', window_size=15):
    
    sr = 16000 # sample rate
    hop_length = 512 # hop length when creating mel_spectrogram
    
    if format == 'waveform':
        length = len(features['audio'])
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

    elif format == 'log-mel-spectrogram': # THIS PART NEEDS TO BE FIXED, SPARSE TENSORS ARE NOT SUBSCRIPTABLE
        slice_length = sr*window_size//hop_length 
        if where == 'middle':
            length = len(features['audio'][1])
            return {
                'audio': tf.sparse.to_dense(features['audio'][:,length-slice_length//2:length+slice_length//2])
                'tid': features['tid']
                'tag': features['tag']}
        elif where == 'beginning':
            return {
                'audio': tf.sparse.to_dense(features['audio'][:,:slice_length])
                'tid': features['tid']
                'tag': features['tag']}

        elif where == 'end':
            return {
                'audio': tf.sparse.to_dense(features['audio'][:,-slice_length:])
                'tid': features['tid']
                'tag': features['tag']}

        elif where == 'random':
            length = len(features['audio'].float_list.value)
            s = np.random.randint(0, length-slice_length)
            return {
                'audio': tf.sparse.to_dense(features['audio'][:,s:s+slice_length])
                'tid': features['tid']
                'tag': features['tag']}
        else:
            print("Please enter a valid location!")
            exit()
    
    return features

def _batch_normalization(features, epsilon=.0001): # not sure if we need this... there's already a batch normalization layer in the model. not even sure if it works
    tensor = tf.unstack(features['spectogram'])
    mean,variance = tf.nn.moments(tensor, axes=[0])
    tensor_normalized = (tensor-mean)/(variance+epsilon)
    
    return {
        'spectogram': tensor_normalized,
        'tid': features['tid'],
        'tags': features['tags']}

def genrate_dataset(root_dir=tfrecord_root_dir, shuffle=True, batch_size=32, buffer_size=10000, window_size=15, reshape=None, with_tags=None, with_tids=None, num_epochs=None):
    if root_dir:
        set_tfrecords_root_dir(os.path.abspath(os.path.expanduser(root_dir)))

    tfrecords = []

    for file in os.listdir(tfrecord_root_dir):
        if file.endswith(".tfrecord"):
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
    
    return dataset.map(lambda x: _window(x, window_size)).batch(batch_size).map(_batch_normalization).repeat(num_epochs)
