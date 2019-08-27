import argparse
import json
import os
from datetime import datetime

import tensorflow as tf

import projectname
import projectname_input

from modules.query_lastfm import LastFm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("api", choices=["built-in", "custom"])
    parser.add_argument("format", choices=["waveform", "log-mel-spectrogram"])
    parser.add_argument("-s", "--split", help="percentage of tracks to go in each dataset, supply as TRAIN VAL TEST", type=int, nargs=3, required=True)
    
    files = parser.add_argument_group('required files')
    files.add_argument("--root-dir", help="directory to read .tfrecord files from, defaults to path on Boden")
    files.add_argument("--config", help="path to config.json, defaults to path on Boden", default='/srv/data/urop/config.json')
    files.add_argument("--lastfm", help="path to (clean) lastfm database, defaults to path on Boden", default='/srv/data/urop/clean_lastfm.db')

    specs1 = parser.add_argument_group('training specs')
    specs1.add_argument("--batch-size", help="specify the batch size during training", type=int, default=32)
    specs1.add_argument("--update_freq", help="specify every what number of batches to write metrics and losses", type=int, default=1)
    specs1.add_argument("--epochs", help="specify the number of epochs to train on", type=int, default=3)
    specs1.add_argument("--steps-per-epoch", help="specify the number of steps to perform for each epoch (if unspecified, go through the whole dataset)", type=int)
    specs1.add_argument("--checkpoint", help="path to previously saved model")

    specs2 = parser.add_argument_group('generating dataset specs')
    specs2.add_argument("--preset", help="use one of the predefined list of tags", default=0)
    specs2.add_argument("--tids", help="list of tids to train on, supply as list of space-separated id's, or as path to .csv file")
    specs2.add_argument("--tags", help="list of tags to train on, supply as list of space-separated tags (spaces within tags need to be escaped)")
    specs2.add_argument("--tags-to-merge", help="list of tags to merge, supply as list of space-separated tags, with '..' as class separator (e.g. '--tags-to-merge rock hard-rock .. hip-hop hip\ hop' merges hard-rock with rock and hip hop with hip-hop)")

    args = parser.parse_args()
    return args
    
def parse_conf(args):

    # load tags database
    lastfm = LastFm(os.path.expanduser(args.lastfm))

    if not os.path.isfile(os.path.expanduser(args.config)):
        args.config = os.path.join(os.path.abspath(os.path.expanduser(args.config)), 'config.json')
    else:
        args.config = os.path.expanduser(args.config)

    # load json
    with open(args.config, 'r') as f:
        d = f.read()
    config = json.loads(d)

    # check whether we are training on a subset of the total tags; sys.argv takes precedence over json
    if args.tags is None:
        args.tags = config['training_options_dataset']['presets'][args.preset]['tags']
    args.tags = lastfm.vec_tag_to_tag_num(args.tags) if args.tags is not None else None # tags might be None in config.json file

    # check whether we are merging any tags; sys.argv takes precedence over json
    if args.tags_to_merge is None:
        args.tags_to_merge = config['training_options_dataset']['presets'][args.preset]['merge_tags']
    args.tags_to_merge = lastfm.tag_to_tag_num(args.tags_to_merge) if args.tags_to_merge is not None else None # merge_tags might be None in config.json file

    # check the total number of tags (that is, output neurons)
    if args.tags is not None:
        args.n_output_neurons = len(args.tags)
    else:
        args.n_output_neurons = config['dataset_specs']['n_tags']
    
    # check model specs
    args.y_input = config['dataset_specs']['n_mels']
    args.n_units = config['model_specs']['n_dense_units']
    args.n_filts = config['model_specs']['n_filters']

    # check train specs
    args.lr = config['training_options']['lr']
    args.mmnt = config['training_options']['momentum']
    args.nest = config['training_options']['nesterov_momentum']

    # check dataset specs
    args.sample_rate = config['dataset_specs']['sample_rate']
    args.shuffle = config['training_options_dataset']['shuffle']
    args.shuffle_buffer = config['training_options_dataset']['shuffle_buffer_size']
    args.window = config['training_options_dataset']['window_length']
    args.window_random = config['training_options_dataset']['window_extract_randomly']

    # if root dir is not specified, use default root dir
    if not args.root_dir:
        if args.sample_rate != 16000:
            s = '-' + str(args.sample_rate // 1000) + 'kHz'
        else:
            s = ''
        args.root_dir = os.path.normpath("/srv/data/urop/tfrecords-" + args.format + s)

    return args

def get_model(args):
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)
        optimizer = tf.keras.optimizers.SGD(lr=args.lr, momentum=args.mmnt, nesterov=args.nest)
        model = projectname.build_model(args.format, args.n_output_neurons, args.y_input, args.n_units, args.n_filts)
        model.compile(loss=loss, optimizer=optimizer, metrics=[[tf.keras.metrics.AUC(curve='ROC', name='AUC-ROC'), tf.keras.metrics.AUC(curve='PR', name='AUC-PR')]])
        if args.checkpoint:
            model.load_weights(os.path.expanduser(config_path))
    
    return model

def train(args, model, train_dataset, valid_dataset):

    log_dir = os.path.expanduser("~/logs/fit/" + datetime.now().strftime("%y%m%d-%H%M"))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath = os.path.join(log_dir, 'mymodel.h5'),
            monitor = 'val_AUC-ROC',
            mode = 'max',
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_AUC-ROC',
            mode = 'max',
            min_delta = 0.2,
            restore_best_weights = True,
            patience = 5,
            verbose = 1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            mode = 'min',
            min_delta = 0.5,
            min_lr = 0.00001,
            factor = 0.2,
            patience = 2,
            verbose = 1,
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir = log_dir, 
            histogram_freq = 1,
            write_graph = False,
            update_freq = args.batch_size*args.update_freq,
            profile_batch = 0,
        ),

        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(train_dataset, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, callbacks=callbacks, validation_data=valid_dataset)

    return history.history

def valid(args, model, test_dataset):
    history = model.evaluate(test_dataset)
    return history

if __name__ == '__main__':
    # argparse
    args = parse_conf(parse_args())

    # create datasets
    train_dataset, valid_dataset, test_dataset = projectname_input.generate_datasets_from_dir(args.root_dir, args.format, args.split, batch_size=32, sample_rate=args.sample_rate, shuffle=args.shuffle, buffer_size=args.shuffle_buffer, window_size=args.window, random=args.window_random, with_tids=args.tids, with_tags=args.tags, merge_tags=args.tags_to_merge, num_epochs=1)
    
    # get model
    model = get_model(args)

    # train
    history = train(args, model, train_dataset, valid_dataset)
    print(history)
