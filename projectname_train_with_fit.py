import json
import os

import numpy as np
import tensorflow as tf

import projectname
import projectname_input
from modules.query_lastfm import LastFm

def main(tfrecords_dir, audio_format, config_path, lastfm_path, split=(70,10,20), preset=0, tids=None, tags=None, tags_to_merge=None, num_epochs=5):

    lastfm = LastFm(os.path.expanduser(lastfm_path))
    
    if not os.path.isfile(os.path.expanduser(config_path)):
        config_path = os.path.join(os.path.abspath(os.path.expanduser(config_path)), 'config.json')
    else:
        config_path = os.path.expanduser(config_path)
    
    with open(config_path, 'r') as f:
        d = f.read()
    config = json.loads(d)

    # check whether we are training on a subset of the tags; check whether we are using default presets, or a custom list of tags
    if tags is None:
        tags = config['training_options_dataset']['presets'][preset]['tags']
    tags = lastfm.vec_tag_to_tag_num(tags) if tags is not None else None # tags might be None in config.json file

    if tags_to_merge is None:
        tags_to_merge = config['training_options_dataset']['presets'][preset]['merge_tags']
    tags_to_merge = lastfm.tag_to_tag_num(tags_to_merge) if tags_to_merge is not None else None # merge_tags might be None in config.json file

    # check the total number of tags (that is, output neurons)
    if tags is not None:
        n_output_neurons = len(tags)
    else:
        n_output_neurons = config['dataset_specs']['n_tags']
    
    # parse model specs
    y_input = config['dataset_specs']['n_mels']
    lr = config['training_options']['lr']
    n_units = config['training_options']['n_dense_units']
    n_filts = config['training_options']['n_filters']

    # generate train and valid datasets
    train_dataset, valid_dataset = projectname_input.generate_datasets_with_split(tfrecords_dir = os.path.expanduser(tfrecords_dir), audio_format = audio_format, split = split, with_tids = tids, with_tags = tags, merge_tags = tags_to_merge, num_epochs=num_epochs)

    # build model
    model = projectname.build_model(audio_format, n_output_neurons, y_input, n_units, n_filts)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr), loss=lambda x, y: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(x, y)), metrics=[[tf.keras.metrics.AUC(curve='ROC', name='roc-auc'), tf.keras.metrics.AUC(curve='PR', name='pr-auc')]])

    history = model.fit(train_dataset, epochs=num_epochs, validation_data=valid_dataset, validation_steps=10)
    
    return history.history

if __name__ == '__main__':
    main('/srv/data/urop/tfrecords-log-mel-spectrogram', 'log-mel-spectrogram', '/srv/data/urop/config.json', '/srv/data/urop/clean_lastfm.db', split=(80,20), num_epochs=5)
    # main('~/Desktop/7D-TFRecords', 'log-mel-spectrogram', '~/Desktop/config.json', '~/Desktop/last.fm/clean_lastfm.db', split=(1,1), num_epochs=5)