import json
import os

import tensorflow as tf

import projectname
import projectname_input

def main(tfrecords_dir, audio_format, config_path, preset=0, tids=None, tags=None, tags_to_merge=None):
    
    if not os.path.isconfig(config_path):
        config_path = os.path.join(os.path.abspath(os.path.expanduser(config_path), 'config.json'))
    
    with open(config_path, 'r') as f:
        d = f.read()
    config = json.loads(d)

    # check whether we are training on a subset of the tags; check whether we are using default presets, or a custom list of tags
    if tags is None:
        tags = config['training_options_dataset']['presets'][preset]['tags']
    if tags_to_merge is None:
        tags_to_merge = config['training_options_dataset']['presets'][preset]['tags_to_merge']
    
    # check the total number of tags (that is, output neurons)
    if tags is not None:
        n_output_neurons = len(tags)
    else:
        n_output_neurons = config['dataset_specs']
    
    # parse model specs
    y_input = config['train_params']['n_mels']
    lr = config['train_params']['lr']
    n_units = config['train_params']['n_dense_units']
    n_filters = config['train_params']['n_filters']

    # generate train and valid datasets
    train_dataset, valid_dataset = projectname_input.generate_dataset_with_split(tfrecords_dir = tfrecords_dir, audio_format = audio_format, train_val_test_split = (80, 20), with_tids = tids, with_tags = tags, merge_tags = tags_to_merge)

    # build model
    model = projectname.build_model(audio_format, n_output_neurons, y_input, n_units, n_filters)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr), loss=lambda x, y: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(x, y)), metrics=[[tf.keras.metrics.AUC(curve='ROC', name='roc-auc'), tf.keras.metrics.AUC(curve='PR', name='pr-auc')]])

    # train
    train()