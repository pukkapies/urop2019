import argparse
import os
import time

import audioread
import librosa
import numpy as np
import tensorflow as tf

from tqdm.auto import tqdm

import lastfm

from data_input import generate_datasets_from_dir
from orpheus_model import build_model
from orpheus_model import parse_config_json

def load_from_checkpoint(audio_format, config, checkpoint_path=None):
    ''' Loads the model from a specified checkpoint.
    
    Parameters
    ----------
    audio_format: {'waveform', 'log-mel-spectrogram'}
        The audio format.

    config: argparse.Namespace()
        The config namespace generated with parse_config_json().

    checkpoint_path: str or None
        The path to the checkpoint to be used (or only to the folder containing it, if latest checkpoint 
        is to be selected). 
        If None, will look for latest checkpoint in the config.json 'log_dir' directory.
    
    Returns
    -------
    model: tf.keras.Model
    '''

    model = build_model(audio_format, num_output_neurons=config.n_output_neurons, num_units=config.n_dense_units, num_filts=config.n_filters, y_input=config.n_mels)
    
    checkpoint_path = checkpoint_path or config.log_dir # if checkpoint_path is not specified, use 'log_dir' from config.json

    # either specified checkpoint, or latest available checkpoint from specified folder
    if os.path.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path).expect_partial()
    print('Restoring checkpoint from {}...'.format(checkpoint_path))
    print()

    return model

def get_audio(mp3_path, audio_format, sample_rate, n_mels=128, array=None, array_sr=None):
    ''' Loads and converts a .mp3 file into the format used by the model. 
    
    Paramters
    ---------
    mp3_path: str
        The path to the .mp3 file to load.

    audio_format: {'waveform', 'log-mel-spectrogram'}
        The audio format.

    sample_rate: int
        The audio (output) sample rate.
        
    array: numpy.ndarray
        If audio is already a waveform array, input audio.
        
    array_sr: int
        If audio is already a waveform array, input audio sample rate.

    Returns
    ------
    np.ndarray
        The processed audio array, converted into the right format.
    '''

    if mp3_path:
        array, sr_in = librosa.core.load(mp3_path, sr=None, mono=False)
    elif array is not None:
        array = array.astype(np.float32)
        sr_in = array_sr
    else:
        raise TypeError("'mp3_path' and 'array' must not both be None")
        
    array = librosa.core.to_mono(array)
    array = librosa.resample(array, sr_in, sample_rate)

    if audio_format == 'log-mel-spectrogram':
        array = librosa.core.power_to_db(librosa.feature.melspectrogram(array, sample_rate, n_mels=n_mels))
        array = array.astype(np.float32)
        mean, variance = tf.nn.moments(tf.constant(array), axes=[0,1], keepdims=True)
        array = tf.nn.batch_normalization(array, mean, variance, offset = 0, scale = 1, variance_epsilon = .000001).numpy()

    return array

def get_audio_slices(audio, audio_format, sample_rate, window_length, n_slices=None):
    ''' Extracts slices of audio along the entire audio array.
    
    Parameters
    ----------
    audio: np.ndarray
        The processed audio array.
    
    audio_format: {'waveform', 'log-mel-spectrogram'}
        The audio format.

    sample_rate: int
        The audio sample rate.

    window_length: int
        The length of the window(s) to be extracted.

    n_slices: int
        The desired number of slices. If None, compute as many slices as possible.

    Returns
    -------
    np.ndarray
        The processed audio array, converted into a batch of 'n_slices' audio windows and ready for being fed into the model.
    '''

    assert audio_format in ('waveform', 'log-mel-spectrogram')
    
    slice_length = window_length * sample_rate // 512 if audio_format == 'log-mel-spectrogram' else window_length * sample_rate

    # compute output shape
    shape = audio.shape[:-1] + (audio.shape[-1] - slice_length + 1, slice_length)
    
    # compute output 'strides' (see https://stackoverflow.com/a/53099870)
    strides = audio.strides + (audio.strides[-1],)
    
    # slice (see https://stackoverflow.com/a/6811241)
    slices = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    
    # transpose (if log-mel-spectrogram)
    if slices.ndim == 3:
        slices = np.transpose(slices, [1, 0, 2]) # want an array of shape (batch_size, *)
    
    # pick 'n_slices' slices from array; or pick them all, if 'n_slices' is None
    n_slices = n_slices or slices.shape[0]
    
    return np.take(slices, np.random.choice(slices.shape[0], size=n_slices, replace=False), axis=0)

def predict(model, fm, audio, config, threshold=0.5):
    ''' Takes a trained model and uses it to predict the tags for a given (batch of) track(s).
    
    Paramters:
    ---------
    model: tf.keras.Model
        Instance of the model to use for predictions.

    fm: LastFm, LastFm2Pandas
        Instance of the tags database.

    audio: np.ndarray
        The processed audio array (or audio 'batch').

    audio_format: {'waveform', 'log-mel-spectrogram'}
        The audio format.

    config: argparse.Namespace
        The namespace generated from config.json with parse_config_json().

    threshold: float
        Only the tag predictions with 'confidence' higher than the threshold will be returned. 

    Returns
    -------
    tags: list
        List of pairs (tag, val) of predicted tags along with their confidence.
    '''

    # compute average by different audio slices
    logits = tf.reduce_mean(model(audio, training=False), axis=[0])
    
    # compute tags
    tags = []
    for idx, val in enumerate(logits):
        if val >= threshold:
            tags.append((fm.tag_num_to_tag(config.tags[idx]), val.numpy()))
            
    # sort predictions from higher confidence to lower confidence
    tags = sorted(tags, key=lambda x: x[1], reverse=True)

    return tags

def test(model, tfrecords_dir, audio_format, config):
    ''' Takes a given model and tests its performance on a test dataset using AUC_ROC and AUC_PR.
    
    Parameters
    ----------
    model: tf.keras.Model
        Instance of the model to test.

    tfrecords_dir: str
        The directory containing the .tfrecord files.

    audio_format: {'waveform', 'log-mel-spectrogram'}
        The audio format.

    config: argparse.Namespace
        The namespace generated from config.json with parse_config_json().
    '''
    _, _, test_dataset = generate_datasets_from_dir(args.tfrecords_dir, args.format, split = config.split, which_split=(True, True, True),
                                                    sample_rate = config.sample_rate, batch_size = config.batch_size, 
                                                    block_length = config.interleave_block_length, cycle_length = config.interleave_cycle_length,
                                                    shuffle = config.shuffle, shuffle_buffer_size = config.shuffle_buffer_size, 
                                                    window_length = config.window_length, window_random = config.window_random, 
                                                    num_mels = config.n_mels, num_tags = config.n_tags, with_tags = config.tags, merge_tags = config.tags_to_merge,
										            as_tuple = True)

    metric_1 = tf.keras.metrics.AUC(name='ROC_AUC',
                                        curve='ROC',
                                        dtype=tf.float32)
    metric_2 = tf.keras.metrics.AUC(name='PR_AUC',
                                        curve='PR',
                                        dtype=tf.float32)

    for entry in tqdm(test_dataset, leave=False):
        audio_batch, label_batch = entry[0], entry[1]
        logits = model(audio_batch, training=False)
        metric_1.update_state(label_batch, logits)
        metric_2.update_state(label_batch, logits)

    print('ROC_AUC: ', np.round(metric_1.result().numpy()*100, 2), '; PR_AUC: ', np.round(metric_2.result().numpy()*100, 2))

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parser = argparse.ArgumentParser()     
    subparsers = parser.add_subparsers(title='commands', dest='mode')
    subparsers.required=True

    testing = subparsers.add_parser('test', help='take a trained model and evaluate its performance on a test dataset')
    testing.add_argument('format', help='model audio format')
    testing.add_argument('--checkpoint', help='path to checkpoint to restore', required=True)
    testing.add_argument('--config', help='path to config.json', required=True)
    testing.add_argument('--lastfm', help='path to (clean) lastfm database (default to path on Boden)', default='/srv/data/urop/clean_lastfm.db')
    testing.add_argument('--tfrecords-dir', help='directory to read the .tfrecord files from (default to path on Boden)')

    predicting = subparsers.add_parser('predict', help='take a trained model and use it to make tag previctions on (single or multiple) .mp3 audio tracks, or on an audio recording')
    predicting.add_argument('format', help='model audio format')
    predicting.add_argument('--checkpoint', help='path to checkpoint to restore', required=True)
    predicting.add_argument('--config', help='path to config.json', required=True)
    predicting.add_argument('--lastfm', help='path to (clean) lastfm database (default to path on Boden)', default='/srv/data/urop/clean_lastfm.db')

    prediction = predicting.add_mutually_exclusive_group(required=True)
    prediction.add_argument('--mp3', dest='mp3_path', help='predict tags using specified .mp3 file (or .mp3 files contained in specified directory)')
    prediction.add_argument('--record', help='predict tags using recorded audio from your microphone', action='store_true')
    
    predicting.add_argument('--record-length', help='length of audio recording (minimum length 15 sec)', type=int, default=15)
    predicting.add_argument('--window-length', help='length (in seconds) of audio window to use for predictions', type=int)
    predicting.add_argument('--n-slices', help='number of slices to use for predictions', type=int, default=20)
    predicting.add_argument('-t', '--threshold', help='threshold for confidence values to count as confident predictions', type=float, default=0.1)

    args = parser.parse_args()

    fm = lastfm.LastFm(args.lastfm)

    config = parse_config_json(args.config, fm)

    model = load_from_checkpoint(args.format, config, checkpoint_path=args.checkpoint) 

    if args.mode == 'test':
        if not args.tfrecords_dir: # if --tfrecords-dir is not specified, use default path on our server
            if config.sample_rate != 16000:
                s = '-' + str(config.sample_rate // 1000) + 'kHz'
            else:
                s = ''
            args.tfrecords_dir = os.path.normpath('/srv/data/urop/tfrecords-' + args.format + s)

        test(model, args.tfrecords_dir, args.format, config)
    
    else:
        args.window_length = args.window_length or config.window_length # if window_length is not specified, use 'window_length' from config.json
        if not args.record:
            if os.path.isfile(args.mp3_path):
                try:
                    narray = get_audio(args.mp3_path, args.format, sample_rate=config.sample_rate, n_mels=config.n_mels)
                    narray = get_audio_slices(narray, args.format, sample_rate=config.sample_rate, window_length=args.window_length, n_slices=args.n_slices)
                    print('Predictions: ', predict(model, fm, narray, config, threshold=args.threshold))
                except audioread.NoBackendError:
                    print('Skipping {} because a NoBackendError occurred...'.format(args.mp3_path))
            else:
                for mp3_path in os.listdir(args.mp3_path): 
                    try:
                        narray = get_audio(mp3_path, args.mp3_path, args.format, sample_rate=config.sample_rate, n_mels=config.n_mels)
                        narray = get_audio_slices(narray, args.format, sample_rate=config.sample_rate, window_length=args.window_length, n_slices=args.n_slices)
                        print('File: ', mp3_path)
                        print('Predictions: ', predict(model, fm, narray, config, threshold=args.threshold))
                    except audioread.NoBackendError:
                        print('Skipping {} because a NoBackendError occurred...'.format(mp3_path))
                        continue
        else:
            assert args.record_length >=15

            import sounddevice as sd  # in case this is not installed automatically
            
            sr_rec = 44100  # sample rate

            while True:
                val = input('Press Enter to begin')
                if val is not None:
                    break

            print('3', end='/r')
            time.sleep(1)
            print('2', end='/r')
            time.sleep(1)
            print('1', end='/r')
            time.sleep(1)
            print('Recording...', end='/r')

            audio = sd.rec(int(args.record_length * sr_rec), samplerate=sr_rec, channels=2)
            sd.wait() # wait until recording is finished
            
            audio = audio.transpose()
            audio = get_audio(mp3_path = None, sample_rate=config.sample_rate, n_mels=config.n_mels, array=audio, array_sr=sr_rec)
            print('Predictions: ', predict(model, audio, config, threshold=args.threshold))
