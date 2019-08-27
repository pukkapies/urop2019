import argparse
import json
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # verbose mode, filter out INFO and WARNING messages 

import tensorflow as tf

import projectname
import projectname_input

from modules.query_lastfm import LastFm
    
def parse_config(path):

    # load tags database
    lastfm = LastFm(os.path.expanduser(args.lastfm))

    if not os.path.isfile(os.path.expanduser(path)):
        path = os.path.join(os.path.abspath(os.path.expanduser(path)), 'config.json')
    else:
        path = os.path.expanduser(path)

    # load json
    with open(path, 'r') as f:
        d = f.read()
    config = json.loads(d)

    # read top tags to use
    top = int(config['training_options']['top'])
    top = set(lastfm.popularity()['tag'][:top].tolist())
    
    tags = top.union(config['training_options']['with_tags'])
    for tag in ['training_options']['without_tags']:
        tags.remove(tag)
    
    config['training_options']['tags'] = lastfm.vec_tag_to_tag_num(list(tags))
    config['training_options']['merge_tags'] = lastfm.tag_to_tag_num(config['training_options']['merge']) if config['training_options']['merge'] is not None else None # merge_tags might be None in config.json file

    # check the total number of tags (that is, output neurons)
    config['n_output_neurons'] = len(config['training_options']['tags'])

    return config

def build_model(args):
    model = projectname.build_model(args.format, args.n_output_neurons, args.y_input, args.n_units, args.n_filts)
    if args.checkpoint:
        model.load_weights(os.path.expanduser(args.checkpoint))
    return model

def build_compiled_model(args):
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)
        optimizer = tf.keras.optimizers.SGD(lr=args.lr, momentum=args.mmnt, nesterov=args.nest)
        model = projectname.build_model(args.format, args.n_output_neurons, args.y_input, args.n_units, args.n_filts)
        model.compile(loss=loss, optimizer=optimizer, metrics=[[tf.keras.metrics.AUC(curve='ROC', name='AUC-ROC'), tf.keras.metrics.AUC(curve='PR', name='AUC-PR')]])
        if args.checkpoint:
            model.load_weights(os.path.expanduser(args.checkpoint))
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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--api", choices=["fit", "custom"], default="fit")
    subparsers = parser.add_subparsers(title="subcommands", dest='command')
    
    train = subparsers.add_parser('train', help="train the model")
    train.add_argument("format", choices=["waveform", "log-mel-spectrogram"])
    train.add_argument("-s", "--perc-split", help="percentage of tracks to go in each dataset, supply as TRAIN VAL (TEST)", type=int, nargs='+')
    
    train_files = train.add_argument_group('paths')
    train_files.add_argument("--root-dir", help="directory to read .tfrecord files from (default to path on Boden)")
    train_files.add_argument("--config", help="path to config.json (default to path on Boden)", default='/srv/data/urop/config.json')
    train_files.add_argument("--lastfm", help="path to (clean) lastfm database (default to path on Boden)", default='/srv/data/urop/clean_lastfm.db')

    train_specs = train.add_argument_group('training options')
    train_specs.add_argument("--epochs", help="specify the number of epochs to train on", type=int, required=True)
    train_specs.add_argument("--steps-per-epoch", help="specify the number of steps to perform for each epoch (if unspecified, go through the whole dataset)", type=int)
    train_specs.add_argument("--update_freq", help="specify the frequency (in steps) to record metrics and losses", type=int, default=10)
    train_specs.add_argument("--checkpoint", help="load a previously saved model")

    evaluate = subparsers.add_parser('evaluate', help="load a previously saved model and evaluate it")
    evaluate.add_argument("format", choices=["waveform", "log-mel-spectrogram"])
    evaluate.add_argument("checkpoint", help="model to load")
    evaluate.add_argument("--root-dir", help="directory to read .tfrecord files from (default to path on Boden)")
    evaluate.add_argument("-s", "--perc-split", help="percentage of tracks to go in each dataset, supply as (TRAIN VAL) TEST", type=int, nargs='+')

    predict = subparsers.add_parser('predict', help="load a previously saved model and predict the tags for a given track")
    predict.add_argument("format", choices=["waveform", "log-mel-spectrogram"])
    predict.add_argument("checkpoint", help="model to load")

    args = parser.parse_args()

    if not args.root_dir:
        if args.sample_rate != 16000:
            s = '-' + str(args.sample_rate // 1000) + 'kHz'
        else:
            s = ''
        args.root_dir = os.path.normpath("/srv/data/urop/tfrecords-" + args.format + s)

    if args.command == 'train':
        # parse config json
        config = parse_config(args.config)

        # generate train and validation dataset
        train_dataset, valid_dataset = projectname_input.generate_datasets_from_dir(tfrecords_dir=args.root_dir, 
                                                                                    audio_format=args.format, 
                                                                                    split=args.perc_split,
                                                                                    sample_rate=config['specs_data']['sample_rate'],
                                                                                    batch_size=config['training_options']['batch_size'],
                                                                                    shuffle=config['training_options']['shuffle'],
                                                                                    buffer_size=config['training_options']['shuffle_buffer_size'],
                                                                                    window_length=config['training_options']['window_len'],
                                                                                    random=config['training_options']['window_slice_random'],
                                                                                    with_tags=config['training_options']['tags'],
                                                                                    merge_tags=config['training_options']['merge_tags'],
                                                                                    num_epochs=1)

        # build model
        model = build_compiled_model(args)

        # train
        train(args, model, train_dataset, valid_dataset)

    elif args.command == 'evaluate':
        pass
    elif args.command == 'predict':
        pass
