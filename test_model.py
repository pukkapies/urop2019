import os
import json

import tensorflow as tf

import modules.query_lastfm as q_fm
import projectname_input
import projectname as Model

def load_from_checkpoint(checkpoint_dir, audio_format, config_dir)
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
                                                            merge_tags=merge_tags, num_tags=155, num_epochs=2)[-1]

    ROC_AUC = tf.keras.metrics.AUC(curve='ROC', name='ROC_AUC',  dtype=tf.float32)
    PR_AUC = tf.keras.metrics.AUC(curve='PR', name='PR_AUC', dtype=tf.float32)

    for entry in dataset:

        audio_batch, label_batch = entry[0], entry[1]

        logits = model(audio_batch)

        ROC_AUC.update_state(label_batch, logits)
        PR_AUC.update_state(label_batch, logits)

    print('ROC_AUC: ', ROC_AUC.result(), '; PR_AUC: ', PR_AUC.result())

def predict(model, audio, with_tags, cutoff, db_path='/srv/data/urop/lastfm_clean.db'):
    ''' Predicts tags given audio for one track '''

    logits = model(audio)
    fm = q_fm.LastFm(db_path)

    tags = []
    
    # if there is only one array
    tags = []

    for idx, probability in enumerate(logits):
        if probability > 0.5:
            tags.append((fm.tag_num_to_tag(with_tags[idx-1]), probability))

    return tags

if __name__ == '__main__':
    # getting tags
    fm = q_fm.LastFm('/srv/data/urop/clean_lastfm.db') 
    tags = fm.popularity().tag.to_list()[:50]
    with_tags = [fm.tag_to_tag_num(tag) for tag in tags]

    model = load_from_ceckpoint('/srv/data/urop/tfrecords-log-mel-spectrogram/', '/home/calle', 'log-mel-spectrogram') 
    # loading model
    test(model, '/srv/data/urop/tfrecords-log-mel-spectrogram/', 'log-mel-spectrogram', (80, 10, 10), batch_size=128, with_tags=with_tags)
