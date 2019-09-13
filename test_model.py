import os
import json
import argparse

import tensorflow as tf
import numpy as np
import librosa

import modules.query_lastfm as q_fm
import projectname_input
import projectname as Model

def parse_config(config_path, lastfm_path):

    # load tags database
    lastfm = q_fm.LastFm(os.path.expanduser(lastfm_path))

    if not os.path.isfile(os.path.expanduser(config_path)):
        path = os.path.join(os.path.abspath(os.path.expanduser(config_path)), 'config.json')
    else:
        path = os.path.expanduser(config_path)

    # load json
    with open(path, 'r') as f:
        config_d = json.loads(f.read())


    n_tags = config_d['tfrecords']['n_tags']
    # read top tags from popularity dataframe
    top = config_d['tags']['top']
    if (top is not None) and (top !=n_tags):
        top_tags = lastfm.popularity()['tag'][:top].tolist()
        tags = set(top_tags)
    else:
        tags=None

    # find tags to use
    if tags is not None:
        if config_d['tags']['with']:
            tags.update(config_d['tags']['with'])

        if config_d['tags']['without']:
            tags.difference_update(config_d['tags']['without'])
    else:
        raise ValueError("parameter 'with' is inconsistent to parameter 'top'")

    # create config namespace (to be accessed more easily than a dictionary)
    config = argparse.Namespace()
    config.batch = config_d['config']['batch_size']
    config.cycle_len = config_d['config']['cycle_length']
    config.early_stop_min_d = config_d['config']['early_stop_min_delta']
    config.early_stop_patience = config_d['config']['early_stop_patience']
    config.n_dense_units = config_d['model']['n_dense_units']
    config.n_filters = config_d['model']['n_filters']
    config.n_mels = config_d['tfrecords']['n_mels']
    config.n_output_neurons = len(tags) if tags is not None else n_tags
    config.path = config_path
    config.plateau_min_d = config_d['config']['reduce_lr_plateau_min_delta']
    config.plateau_patience = config_d['config']['reduce_lr_plateau_patience']
    config.shuffle = config_d['config']['shuffle']
    config.shuffle_buffer = config_d['config']['shuffle_buffer_size']
    config.split = config_d['config']['split']
    config.sr = config_d['tfrecords']['sample_rate']
    config.tags = lastfm.vec_tag_to_tag_num(list(tags)) if tags is not None else None
    config.tags_to_merge = lastfm.vec_tag_to_tag_num(config_d['tags']['merge']) if config_d['tags']['merge'] is not None else None
    config.tot_tags = config_d['tfrecords']['n_tags']
    config.window_len = config_d['config']['window_length']
    config.window_random = config_d['config']['window_extract_randomly']
    config.log_dir = config_d['config']['log_dir']
    config.checkpoint_dir = config_d['config']['checkpoint_dir']

    # create config namespace for the optimizer (will be used by get_optimizer() in order to allow max flexibility)
    config_optim = argparse.Namespace()
    config_optim.class_name = config_d['optimizer'].pop('name')
    config_optim.config = config_d['optimizer']

    return config, config_optim

def load_from_checkpoint(audio_format, config, checkpoint_path=None):
    ''' Loads model from checkpoint '''

    # loading model
    model = Model.build_model(frontend_mode=audio_format, 
                                num_output_neurons=config.n_output_neurons, y_input=config.n_mels,
                                num_units=config.n_dense_units, num_filts=config.n_filters)
    
    # restoring from checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    if checkpoint_path:
        print('Loading from {}'.format(checkpoint_path))
        checkpoint.restore(checkpoint_path)
    else:
        # loading latest training checkpoint 
        latest = tf.train.latest_checkpoint(config.checkpoint_dir)
        print('Loading from {}'.format(latest))
        checkpoint.restore(latest)

    return model

def get_audio(mp3_path, audio_format, config):

    array, sr_in = librosa.core.load(mp3_path, sr=None, mono=False)
    array = librosa.core.to_mono(array)
    array = librosa.resample(array, sr_in, config.sr)

    if audio_format == "log-mel-spectrogram":
        array = librosa.core.power_to_db(librosa.feature.melspectrogram(array, config.sr, n_mels=config.n_mels))
        # normalization
        mean, variance = tf.nn.moments(tf.constant(array), axes=[0,1], keepdims=True)
        array = (array - mean) / np.sqrt(variance+0.000001)

    return array

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

        logits = tf.multiply(model(audio_batch), tf.constant(0.001, dtype=tf.float32))

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
    # make sure tags are sorted
    with_tags = np.sort(with_tags)

    # compute average by using a moving window
    slices = get_slices(audio, audio_format, sample_rate, window_size)
    logits = tf.reduce_mean(model(slices, training=False), axis=[0])
    
    # get tags
    tags = []
    for idx, val in enumerate(logits):
        if val >= cutoff:
            tags.append([float(val.numpy()), fm.tag_num_to_tag(int(with_tags[idx]))])
    return tags

if __name__ == '__main__':

    parser = argparse.ArgumentParser()     
    parser.add_argument("config-path", help="Path to config JSON file")
    parser.add_argument("checkpoint", help="Path to a checkpoint or directory of checkpoints")
    parser.add_argument("audio-format", help="Model audio format")
    parser.add_argument("mode", choices=["predict", "test"], help="Choose functionality of script, testing or predict")
    parser.add_argument("--lastfm-path", help="Path to lastfm database", default="/home/calle/clean_lastfm.db")
    parser.add_argument("--tfrecords-dir", help="Path to tfrecords directory, specify if test mode has been selected")
    parser.add_argument("--mp3-path", help="Path to mp3 dir or mp3 file to predict")
    parser.add_argument("--cutoff", help="Lower bound for what prediction values to print")

    args = parser.parse_args()

    config = parse_config(args.config_path, args.lastfm_path)[0]
    model = load_from_checkpoint(args.audio_format, config, checkpoint_path=args.checkpoint) 

    if args.mode == "test":
        test(model, args.tfrecords_dir, args.audio_format, config.split, batch_size=config.batch_size, with_tags=config.tags)
    else:
        if not args.mp3_path:
            raise ValueError("If predicting, must specify mp3 file(s) to predict")
        elif os.path.isfile(args.mp3_path):
            audio = get_audio(args.mp3_path, args.audio_format, args.config)
            print("prediction: ", predict(model, audio, args.audio_format, config.tags, config.sr, cutoff=args.cutoff, db_path=args.lastfm_path))
        else:
            for path in os.listdir(args.mp3_path): 
                audio = get_audio(path, args.audio_format, config)
                print("file: ", path)
                print(̈́"prediction: ", predict(model, audio, args.audio_format, config.tags, config.sr, cutoff=args.cutoff, db_path=args.lastfm_path))
