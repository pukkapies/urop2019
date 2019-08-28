import os
import json

import tensorflow as tf
import numpy as np

import modules.query_lastfm as q_fm
import projectname_input
import projectname as Model

def load_from_checkpoint(checkpoint_dir, audio_format, config_dir):
    ''' Loads model from checkpoint '''

    # getting config settings
    if not os.path.isfile(config_dir):
        config_dir = os.path.join(os.path.normpath(config_dir), 'config.json')

    with open(config_dir) as f:
        file = json.load(f)

    num_tags = file['dataset_specs']['n_tags']
    y_input = file['dataset_specs']['n_mels']
    num_units = file['training_options']['n_dense_units']
    num_filt = file['training_options']['n_filters']

    # loading model
    model = Model.build_model(frontend_mode=audio_format, 
                                num_output_neurons=num_tags, y_input=y_input,
                                num_units=num_units, num_filt=num_filt)

    # loading latest training checkpoint 
    checkpoint = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('Loadind from {}'.format(latest))
    checkpoint.restore(latest)
    return model

def test(model, tfrecords_dir, audio_format, split, batch_size=64, window_size=15, random=False, with_tags=None, with_tids=None, merge_tags=None):
    ''' Tests model '''

    # loading test dataset
    dataset = projectname_input.generate_datasets_from_dir(tfrecords_dir, audio_format, split=split, 
                                                            batch_size=batch_size, shuffle=False, window_size=window_size,
                                                            random=random, with_tags=with_tags, with_tids=with_tids,
                                                            merge_tags=merge_tags, num_tags=155, num_epochs=1)[-1]

    ROC_AUC = tf.keras.metrics.AUC(curve='ROC', name='ROC_AUC',  dtype=tf.float32)
    PR_AUC = tf.keras.metrics.AUC(curve='PR', name='PR_AUC', dtype=tf.float32)

    for entry in dataset:

        audio_batch, label_batch = entry[0], entry[1]

        logits = model(audio_batch)

        ROC_AUC.update_state(label_batch, logits)
        PR_AUC.update_state(label_batch, logits)

    print('ROC_AUC: ', int(ROC_AUC.result()), '; PR_AUC: ', int(PR_AUC.result()))

def get_slices(audio, audio_format, sample_rate, window_size=15):
    
    if audio_format == 'waveform':
        slice_length = window_size*sample_rate
        n_slices = audio.shape[0]//slice_length
        slices = [audio[i*slice_length:(i+1)*slice_length] for i in range(n_slices)] 
        slices.append(audio[-slice_length:])
        return np.array(slices)

    elif audio_format == 'log-mel-spectrogram':
        slice_length = window_size*sample_rate//512
        n_slices = audio.shape[1]//slice_length

        slices = [audio[:,i*slice_length:(i+1)*slice_length] for i in range(n_slices)]
        slices.append(audio[:,-slice_length:])
        return np.array(slices)

def predict(model, audio, audio_format, with_tags, sample_rate, cutoff=0.5, window_size=15, db_path='/srv/data/urop/clean_lastfm.db'):
    ''' Predicts tags given audio for one track '''

    fm = q_fm.LastFm(db_path)

    # compute average by using a moving window
    slices = get_slices(audio, audio_format, sample_rate, window_size)
    logits = tf.reduce_mean(model(slices, training=False), axis=[0])
    
    # get tags
    tags = []
    for idx, val in enumerate(logits):
        if val >= cutoff:
            tags.append((fm.tag_num_to_tag(with_tags[idx]), val.numpy()))
    return tags

if __name__ == '__main__':
    # getting tags
    fm = q_fm.LastFm('/srv/data/urop/clean_lastfm.db') 
    tags = fm.popularity().tag.to_list()[:50]
    with_tags = [fm.tag_to_tag_num(tag) for tag in tags]

    model = load_from_checkpoint('/srv/data/urop/model_aden/log-mel-spectrogram_20190826-103644/', 'log-mel-spectrogram', '/home/calle') 
    # loading model
    # test(model, '/srv/data/urop/tfrecords-log-mel-spectrogram/', 'log-mel-spectrogram', (80, 10, 10), batch_size=128, with_tags=with_tags)

    AUDIO_FEATURES_DESCRIPTION = {
        'audio': tf.io.VarLenFeature(tf.float32),
        'tid': tf.io.FixedLenFeature((), tf.string),
        'tags': tf.io.FixedLenFeature((155, ), tf.int64)
    }


    dataset = tf.data.TFRecordDataset('/srv/data/urop/tfrecords-log-mel-spectrogram/log-mel-spectrogram_1.tfrecord')
    dataset = dataset.map(lambda x: projectname_input._parse_features(x, AUDIO_FEATURES_DESCRIPTION, (96, -1)))
    i = 0 
    for entry in dataset.take(3):
        if i <= 1:
            i += 1
            continue
        tid = entry['tid'].numpy().decode('utf-8')
        print('TID: ', tid)
        print('tags: ', [tag for tag in fm.query_tags(tid) if tag in tags])
        print('predicted tags: ', predict(model, entry['audio'], 'log-mel-spectrogram', with_tags, 16000, cutoff=0.0001))
