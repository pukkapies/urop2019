import argparse
import datetime
import json
import os

import numpy as np
import tensorflow as tf

import projectname
import projectname_input
from modules.query_lastfm import LastFm

def build_compiled_model(audio_format, n_output_neurons, y_input, n_units, n_filts):
    OPTIMIZER = tf.keras.optimizers.SGD(lr=0.001)
    LOSS = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)
    METRICS = [[tf.keras.metrics.AUC(curve='ROC', name='roc-auc'), tf.keras.metrics.AUC(curve='PR', name='pr-auc')]]

    model = projectname.build_model(audio_format, n_output_neurons, y_input, n_units, n_filts)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    return model

def train(model, train_dataset, valid_dataset=None, num_epochs=5, num_steps_per_epoch=None),:

    log_dir="logs/fit/" + datetime.datetime.now().strftime("%y%m%d-%H%M")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='mymodel_{epoch}.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=1),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, 
            histogram_freq=1),
        ]

    history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=num_steps_per_epoch, callbacks=callbacks, validation_data=valid_dataset)

    return history.history

def main(tfrecords_dir, audio_format, config_path, lastfm_path, preset=0, batch_size=32, shuffle=False, window_length=15, tids=None, tags=None, tags_to_merge=None, num_epochs=None, num_steps_per_epoch=None):

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

    # build model
    model = build_compiled_model(audio_format, n_output_neurons, y_input, n_units, n_filts)

    return model

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-f", "--format", help="set output format of audio, defaults to waveform (e.g. 'log-mel-spectrogram')")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--split", nargs='+', type=int, help="the proportion of TRAIN, VAL, TEST dictates how many entries is in each file, supply as list of space-separated values")
    parser.add_argument("--root-dir", help="directory to read .tfrecord files from")
    parser.add_argument("--config", help="path to config.json")
    parser.add_argument("--lastfm", help="path to (clean) lastfm database")
    parser.add_argument("--tids", help="list of tids to train on, supply as list (separated by /), or as path to .csv file")
    parser.add_argument("--tags", help='list of tags to train on, supply as list (separated by /)')
    parser.add_argument("--tags-to-merge", help="list of tags to merge, supply list of space-separated lists, with '/'-separated values")
    parser.add_argument("--batch-size", type=int, default=32, help="specify the batch size during training")
    parser.add_argument("--shuffle", type=bool, default=False, help="specify whether to shuffle the dataset")
    parser.add_argument("--window-length", type=int, default=15, help="specify the length (in seconds) of the extracted window")
    parser.add_argument("--epochs", type=int, default=1, help="specify the number of epochs to train on")
    parser.add_argument("--steps-per-epoch", type=int, help="specify the number of steps per epoch to train on (if n_epochs not specified)")

    args = parser.parse_args()