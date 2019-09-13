import argparse
import json
import os

from tensorflow import distribute

import projectname_input

from lastfm import LastFm

def parse_config(config_path, lastfm_path):

    # load tags database
    lastfm = LastFm(os.path.expanduser(lastfm_path))

    # if config_path is a folder, assume the folder contains a config.json
    if os.path.isdir(os.path.expanduser(config_path)):
        path = os.path.join(os.path.abspath(os.path.expanduser(config_path)), 'config.json')
    else:
        path = os.path.expanduser(config_path)

    # load json
    with open(path, 'r') as f:
        config_dict = json.loads(f.read())

    # create config namespace
    config = argparse.Namespace(**config_dict['model'], **config_dict['model-training'], **config_dict['tfrecords'])
    config.path = os.path.abspath(config_path)

    # update config (optimizer will be instantiated with tf.get_optimizer using {"class_name": config.optimizer_name, "config": config.optimizer})
    config.optimizer_name = config.optimizer.pop('name')

    # read tags from popularity dataframe
    top = config_dict['tags']['top']
    if top is None:
        top = config.n_tags
    tags = set(lastfm.popularity()['tag'][:top].tolist())

    # read tags to add or discard, and update the tags set
    if config_dict['tags']['with']:
        tags.update(config_dict['tags']['with'])
    if config_dict['tags']['without']:
        tags.difference_update(config_dict['tags']['without'])
    tags = list(tags)

    config.n_output_neurons = len(tags)
    config.tags = lastfm.vec_tag_to_tag_num(tags)
    config.tags_to_merge = lastfm.vec_tag_to_tag_num(config_dict['tags']['merge']) if config_dict['tags']['merge'] else None
    
    return config

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('frontend', choices=['waveform', 'log-mel-spectrogram'])
    parser.add_argument('--root-dir', dest='tfrecords_dir', help='directory to read the .tfrecord files from (default to path on Boden)')
    parser.add_argument('--config-path', help='path to config.json (default to path on Boden)', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json'))
    parser.add_argument('--lastfm-path', help='path to (clean) lastfm database (default to path on Boden)', default='/srv/data/urop/clean_lastfm.db')
    parser.add_argument('--multi-db', help='specify the number of different tags features in the .tfrecord files', type=int, default=1)
    parser.add_argument('--multi-db-default', help='specify the index of the default tags database, when there are more than one tags features in the .tfrecord files', type=int)
    parser.add_argument('--epochs', help='specify the number of epochs to train on', type=int, required=True)
    parser.add_argument('--steps-per-epoch', help='specify the number of steps to perform for each epoch (if unspecified, go through the whole dataset)', type=int)
    parser.add_argument('--no-shuffle', action='store_true', help='force no shuffle, override config setting')
    parser.add_argument('--resume', help='load a previously saved model')
    parser.add_argument('--update-freq', help='specify the frequency (in steps) to record metrics and losses', type=int, default=10)
    parser.add_argument('--cuda', help='set cuda visible devices', type=int, nargs='+')
    parser.add_argument('--custom', action='store_true', help='train using custom training loop')
    parser.add_argument('-v', '--verbose', choices=['0', '1', '2', '3'], help='verbose mode', default='0')

    args = parser.parse_args()

    # import the right training function (either custom loop, or built-in)
    if args.custom:
        from training_gradtape import train
    else:
        from training import train

    # specify number of visible gpu's
    if args.cuda:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # see issue #152
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.cuda])

    # specify verbose mode
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.verbose

    # parse config
    config = parse_config(args.config_path, args.lastfm_path)

    # if root_dir is not specified, use default path on our server
    if not args.tfrecords_dir:
        if config.sample_rate != 16000:
            s = '-' + str(config.sr // 1000) + 'kHz'
        else:
            s = ''
        args.tfrecords_dir = os.path.normpath('/srv/data/urop/tfrecords-' + args.frontend + s)

    # override config setting
    if args.no_shuffle:
        config.shuffle = False

    # generate train_dataset and valid_dataset (valid_dataset will be None if config.split is None)
    train_dataset, valid_dataset = projectname_input.generate_datasets_from_dir(args.tfrecords_dir, args.frontend, split = config.split, which_split=(True, True, ) + (False, ) * (len(config.split)-2),
                                                                                sample_rate = config.sample_rate, batch_size = config.batch_size, 
                                                                                block_length = config.interleave_block_length, cycle_length = config.interleave_cycle_length,
                                                                                shuffle = config.shuffle, shuffle_buffer_size = config.shuffle_buffer_size, 
                                                                                window_length = config.window_length, window_random = config.window_random, 
                                                                                num_mels = config.n_mels, num_tags = config.n_tags, num_tags_db = args.multi_db, default_tags_db = args.multi_db_default, with_tags = config.tags, merge_tags = config.tags_to_merge,
										                                        as_tuple = True)

    # set up training strategy
    strategy = distribute.MirroredStrategy()

    if args.custom:
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        if valid_dataset is not None:
            valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)

    # train
    train(train_dataset, valid_dataset, frontend=args.frontend,
          strategy=strategy, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, 
          config=config,
          update_freq=args.update_freq, timestamp_to_resume=args.resume)
