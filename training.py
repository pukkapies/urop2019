import argparse
import datetime
import json
import os

import tensorflow as tf

import projectname
import projectname_input
from modules.query_lastfm import LastFm

def _required_length(nmin, nmax):
        class RequiredLength(argparse.Action):
            def __call__(self, parser, args, values, option_string=None):
                if not nmin<=len(values)<=nmax:
                    msg='argument "{f}" requires between {nmin} and {nmax} arguments'.format(f=self.dest, nmin=nmin, nmax=nmax)
                    raise argparse.ArgumentTypeError(msg)
                setattr(args, self.dest, values)
        return RequiredLength

def parse_config(config_path, lastfm_path):

    # load tags database
    lastfm = LastFm(os.path.expanduser(lastfm_path))

    if not os.path.isfile(os.path.expanduser(config_path)):
        path = os.path.join(os.path.abspath(os.path.expanduser(config_path)), 'config.json')
    else:
        path = os.path.expanduser(config_path)

    # load json
    with open(path, 'r') as f:
        config_d = json.loads(f.read())

    # read top tags from popularity dataframe
    top = int(config_d['tags']['top'])
    top_tags = lastfm.popularity()['tag'][:top].tolist()
    tags = set(top_tags)

    # find tags to use
    if config_d['tags']['with']:
        tags.union(config_d['tags']['with'])
    if config_d['tags']['without']:
        tags.discard(config_d['tags']['without'])

    # create config namespace (to be accessed more easily than a dictionary)
    config = argparse.Namespace()
    config.batch = config_d['config']['batch_size']
    config.cycle_len = config_d['config']['cycle_length']
    config.early_stop_min_d = config_d['config']['early_stop_min_delta']
    config.early_stop_patience = config_d['config']['early_stop_patience']
    config.n_dense_units = config_d['model']['n_dense_units']
    config.n_filters = config_d['model']['n_filters']
    config.n_mels = config_d['tfrecords']['n_mels']
    config.n_output_neurons = len(tags)
    config.path = config_path
    config.plateau_min_d = config_d['config']['reduce_lr_plateau_min_delta']
    config.plateau_patience = config_d['config']['reduce_lr_plateau_patience']
    config.shuffle = config_d['config']['shuffle']
    config.shuffle_buffer = config_d['config']['shuffle_buffer_size']
    config.split = config_d['config']['split']
    config.sr = config_d['tfrecords']['sample_rate']
    config.tags = lastfm.vec_tag_to_tag_num(list(tags))
    config.tags_to_merge = lastfm.tag_to_tag_num(config_d['tags']['merge']) if config_d['tags']['merge'] is not None else None
    config.tot_tags = config_d['tfrecords']['n_tags']
    config.window_len = config_d['config']['window_length']
    config.window_random = config_d['config']['window_extract_randomly']
    config.log_dir: config_d['config']['log_dir']
    config.checkpoint_dir: config_d['config']['checkpoint_dir']

    # create config namespace for the optimizer (will be used by get_optimizer() in order to allow max flexibility)
    config_optim = argparse.Namespace()
    config_optim.class_name = config_d['optimizer'].pop('name')
    config_optim.config = config_d['optimizer']
    
    return config, config_optim

def get_compiled_model(frontend, strategy, config, config_optim, resume_time=None):
    with strategy.scope():
        # read optimizer specs from config_optim dict for max flexibility
        optimizer = tf.keras.optimizers.get({"class_name": config_optim.class_name, "config": config_optim.config})

        # compile model
        model = projectname.build_model(frontend, num_output_neurons=config.n_output_neurons, num_units=config.n_dense_units, num_filts=config.n_filters, y_input=config.n_mels)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM), metrics=[[tf.keras.metrics.AUC(curve='ROC', name='AUC-ROC'), tf.keras.metrics.AUC(curve='PR', name='AUC-PR')]])
        
        # restore resume_time (if provided)
        if resume_time:
            model.load_weights(os.path.expanduser(resume_time))
    return model

def train(train_dataset, valid_dataset, frontend, strategy, config, config_optim, epochs, steps_per_epoch=None, resume_time=None, update_freq=1, profile_batch=0):

    model = get_compiled_model(frontend, strategy, config, config_optim, resume_time)

    log_dir = os.path.join(os.path.expanduser(config.log), frontend, datetime.datetime.now().strftime("%y%m%d-%H%M")) # to access training scalars using tensorboard

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
            min_delta = config.early_stop_min_d,
            restore_best_weights = True,
            patience = config.early_stop_patience,
            verbose = 1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            mode = 'min',
            min_delta = config.plateau_min_d,
            min_lr = 0.00001,
            factor = 0.2,
            patience = config.plateau_patience,
            verbose = 1,
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir = log_dir, 
            histogram_freq = 1,
            write_graph = False,
            update_freq = update_freq,
            profile_batch = profile_batch, # make sure the variable LD_LIBRARY_PATH is properly set up
        ),

        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, validation_data=valid_dataset)

    return history.history

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("frontend", choices=["waveform", "log-mel-spectrogram"])
    parser.add_argument("--root-dir", dest="tfrecords_dir", help="directory to read .tfrecord files from (default to path on Boden)")
    parser.add_argument("--path-config", help="path to config.json (default to path on Boden)", default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json'))
    parser.add_argument("--path-lastfm", help="path to (clean) lastfm database (default to path on Boden)", default="/srv/data/urop/clean_lastfm.db")
    parser.add_argument("--epochs", help="specify the number of epochs to train on", type=int, required=True)
    parser.add_argument("--steps-per-epoch", help="specify the number of steps to perform for each epoch (if unspecified, go through the whole dataset)", type=int)
    parser.add_argument("--resume-time", help="load a previously saved model")
    parser.add_argument("--update-freq", help="specify the frequency (in steps) to record metrics and losses", type=int, default=10)
    parser.add_argument("--cuda", help="set cuda visible devices", type=int, nargs="+")
    parser.add_argument("-v", "--verbose", choices=['0', '1', '2', '3'], help="verbose mode", default='2')

    args = parser.parse_args()

    # specify number of visible gpu's
    if args.cuda:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.cuda])

    # specify verbose mode
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.verbose

    # parse json
    config, config_optim = parse_config(args.path_config, args.path_lastfm)

    # if root_dir is not specified, use default path on our server
    if not args.tfrecords_dir:
        if config.sr != 16000:
            s = '-' + str(config.sr // 1000) + 'kHz'
        else:
            s = ''
        args.tfrecords_dir = os.path.normpath("/srv/data/urop/tfrecords-" + args.frontend + s)

    # create training and validation datasets
    assert config.split
    assert len(config.split) >= 2
    assert len(config.split) <= 3
    train_dataset, valid_dataset = projectname_input.generate_datasets_from_dir(args.tfrecords_dir, args.frontend, split=args.split, which_split=(True, True, ) + (False, ) * (len(args.split)-2),
                                                                                sample_rate=config.sr, batch_size=config.batch, 
                                                                                cycle_length=config.cycle_len, 
                                                                                shuffle=config.shuffle, buffer_size=config.shuffle_buffer, 
                                                                                num_tags=config.tot_tags, window_size=config.window_len, random=config.window_random, 
                                                                                with_tags=config.tags, merge_tags=config.tags_to_merge)

    # set up training strategy
    strategy = tf.distribute.MirroredStrategy()

    # train
    train(train_dataset, valid_dataset, frontend=args.frontend, strategy=strategy,
          config=config, config_optim=config_optim,
          epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, resume_time=args.resume_time, 
          update_freq=args.update_freq)