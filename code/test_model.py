import argparse
import os
import time

import audioread
import librosa
import numpy as np
import tensorflow as tf

import lastfm
import projectname_input
import projectname

from projectname_train import parse_config

def load_from_checkpoint(audio_format, config, checkpoint_path=None):
    ''' Loads the model from a specified checkpoint.
    
    Parameters
    ----------
    audio_format: {'waveform', 'log-mel-spectrogram'}
        The audio format.

    config: argparse.Namespace()
        The config namespace generated with parse_config().

    checkpoint_path: str or None
        The path to the checkpoint to be used (or only to the folder containing it, if latest checkpoint 
        is to be selected). 
        If None, will look for latest checkpoint in the config.json 'log_dir' directory.
    
    Returns
    -------
    model: tf.keras.Model
    '''

    model = projectname.build_model(audio_format, num_output_neurons=config.n_output_neurons, num_units=config.n_dense_units, num_filts=config.n_filters, y_input=config.n_mels)
    
    checkpoint_path = checkpoint_path or config.log_dir # if checkpoint_path is not specified, use 'log_dir' from config.json

    # either specified checkpoint, or latest available checkpoint from specified folder
    if os.path.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path).expect_partial()
    print('Loading from {}'.format(checkpoint_path))

    return model

def get_audio(mp3_path, audio_format, sample_rate, array=None, array_sr=None):
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
        
    array = librosa.core.to_mono(array)
    array = librosa.resample(array, sr_in, sample_rate)

    if audio_format == "log-mel-spectrogram":
        array = librosa.core.power_to_db(librosa.feature.melspectrogram(array, config.sample_rate, n_mels=config.n_mels))
        array = array.astype(np.float32)
        # normalization
        mean, variance = tf.nn.moments(tf.constant(array), axes=[0,1], keepdims=True)
        array = tf.nn.batch_normalization(array, mean, variance, offset = 0, scale = 1, variance_epsilon = .000001).numpy()

    return array

def get_audio_slices(audio, audio_format, sample_rate, window_length, n_slices=None):
    ''' Extracts slices of audio along the entire audio array.
    
    Parameters
    ----------
    audio :
        The processed audio array.
    
    audio_format: {'waveform', 'log-mel-spectrogram'}
        The audio format.

    sample_rate: int
        The audio sample rate.

    window_length: int
        The length of the window(s) to be extracted.

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

    audio :
        The processed audio array (or audio 'batch').

    audio_format: {'waveform', 'log-mel-spectrogram'}
        The audio format.

    config: argparse.Namespace
        The namespace generated from config.json with parse_config().

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
            tags.append((fm.tag_num_to_tag(config.with_tags[idx]), val.numpy()))
            
    # sort predictions from higher confidence to lower confidence
    tags = sorted(tags, key=lambda x: x[1], reverse=True)

    return tags

def test(model, tfrecords_dir, audio_format, split, batch_size=64, window_length=15, merge_tags=None, window_random=False, with_tags=None, with_tids=None, num_tags=155):
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
        The namespace generated from config.json with parse_config().
    '''
    _, _, test_dataset = projectname_input.generate_datasets_from_dir(args.tfrecords_dir, args.frontend, split = config.split, which_split=(True, True, ) + (False, ) * (len(config.split)-2),
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

    for entry in test_dataset:
        audio_batch, label_batch = entry[0], entry[1]
        logits = model(audio_batch, training=False)
        metric_1.update_state(label_batch, logits)
        metric_2.update_state(label_batch, logits)

    print('ROC_AUC: ', np.round(metric_1.result().numpy()*100, 2), '; PR_AUC: ', np.round(metric_2.result().numpy()*100, 2))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()     
    parser.add_argument("format", help="Model audio format")
    parser.add_argument("mode", choices=["predict", "test"], help="Choose functionality of script, testing or predict")
    parser.add_argument("config", help="Path to config JSON file")
    parser.add_argument("--checkpoint", help="Path to a checkpoints, will default to directory in config.")
    parser.add_argument("--lastfm-path", help="Path to lastfm database", default="/srv/data/urop/clean_lastfm.db")
    parser.add_argument("--tfrecords-dir", help="Path to tfrecords directory, specify if test mode has been selected")
    parser.add_argument("--mp3-path", help="Path to mp3 dir or mp3 file to predict")
    parser.add_argument("--from-recording", help="If True, the input audio will be recorded from your microphone", action="store_true")
    parser.add_argument("-s", "--recording-second", help="Number of seconds to record. Minimum length is 15 seconds", type=int, default='15')
    parser.add_argument("--threshold", type=float, help="Lower bound for what prediction values to print", default=0.1)

    args = parser.parse_args()
    print(args)

    config = parse_config(args.config, args.lastfm_path)
    model = load_from_checkpoint(args.format, config, checkpoint_path=args.checkpoint) 
    print(type(model))

    if args.mode == "test":
        test(model, args.tfrecords_dir, args.format, config.split, batch_size=config.batch_size, window_random=config.window_random, with_tags=config.tags, merge_tags=config.tags_to_merge, num_tags=config.n_tags)
    else:
        
        if not (args.mp3_path or args.from_recording):
            raise ValueError("If predicting, must either specify mp3 file(s) to predict, or set --from-recording as True")
        elif (args.mp3_path and args.from_recording):
            raise ValueError("If predicting, must either specify mp3 file(s) to predict, or set --from-recording as True")
        elif args.mp3_path:
            if os.path.isfile(args.mp3_path):
                try:
                    audio = get_audio(audio_format=args.format, sample_rate=config.sample_rate, mp3_path=args.mp3_path)
                    print("prediction: ", predict(model, audio, audio_format=args.format, with_tags=config.tags, sample_rate=config.sample_rate, threshold=args.threshold, window_length=config.window_length))
                except audioread.NoBackendError:
                    print('skipping {} due to NoBackendError.'.format(args.mp3_path))
                
            else:
                for path in os.listdir(args.mp3_path): 
                    try:
                        audio = get_audio(audio_format=args.format, sample_rate=config.sample_rate, mp3_path=os.path.join(args.mp3_path, path))
                    except audioread.NoBackendError:
                        print('skipping {} due to NoBackendError.'.format(path))
                        continue
                
                    print("file: ", path)
                    print("prediction: ", predict(model, audio, audio_format=args.format, with_tags=config.tags, sample_rate=config.sample_rate, threshold=args.threshold, window_length=config.window_length))
                    print()
                
        else:
            assert args.recording_second >=15
            
            # In case this is not installed automatically
            import sounddevice as sd
            sr_rec = 44100  # Sample rate
            seconds = int(args.second)  # Duration of recording

            while True:
                val = input('Press Enter to begin')
                if val is not None:
                    break

            print('record starts in')
            print('3')
            time.sleep(1)
            print('2')
            time.sleep(1)
            print('1')
            time.sleep(1)
            print('0')
            print('Recording')
            audio = sd.rec(int(seconds * sr_rec), samplerate=sr_rec, channels=2)
            sd.wait()  # Wait until recording is finished

            audio = sd.rec(int(seconds * sr_rec), samplerate=sr_rec, channels=2)
            sd.wait()  # Wait until recording is finished
            
            audio = audio.transpose()
            audio = get_audio(sample_rate=config.sample_rate, array=audio, array_sr=sr_rec)
            print("prediction: ", predict(model, audio, args.format, config.tags, sr_rec, threshold=args.threshold, db_path=args.lastfm_path))
            print("prediction: ", predict(model, audio, args.format, config.tags, config.sample_rate, threshold=args.threshold, db_path=args.lastfm_path))
            
            
            
