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
    audio_format : {'waveform', 'log-mel-spectrogram'}
        The audio format.

    config : argparse.Namespace()
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
    mp3_path : str
        The path to the .mp3 file to load.

    audio_format : {'waveform', 'log-mel-spectrogram'}
        The audio format.

    sample_rate : int
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
    
    audio_format : {'waveform', 'log-mel-spectrogram'}
        The audio format.

    sample_rate : int
        The audio sample rate.

    window_length : int
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

def test(model, tfrecords_dir, audio_format, split, batch_size=64, window_length=15, merge_tags=None, window_random=False, with_tags=None, with_tids=None, num_tags=155):
    ''' Tests a given model with respect to the metrics AUC_ROC and AUC_PR
    
    Parameters
    ----------
    model : tf.keras.Model object
        model to test

    tfrecords_dir : str
        path to directory containing TFRecord files

    audio_format : {'waveform', 'log-mel-spectrogram'}
        audio format used in model

    split : list of three floats
        number of (or percentage of) .tfrecord files that will go in each train/validation/test dataset (ideally an array of len <= 3).
        Note that in this function, only the parameter for test will depend on the size of the tfrecords that will be fed into this function,
        while the train parameter and the validation parameter will decide the position of the test parameter.
        E.g. split = [80,10,10] means the last 10% will be fed into the function.

    batch_size : int

    window_length : int
        size in seconds of window to be extracted from each audio array
        
    merge_tags : list of list of int
        e.g. [[1,2], [2,3]] means merge 1 with 2, and 2 with 3 respectively

    window_random : bool
        Specifies if windows should be extracted from a random location, or from the center of the array

    with_tags : list
        list of tags used during training 

    with_tids : list
        list of tids used during training
        
    num_tags : int
        number of tags in the clean lastfm database
        
    '''

    # loading test dataset
    dataset = projectname_input.generate_datasets_from_dir(tfrecords_dir, audio_format, split=split, 
                                                            batch_size=batch_size, shuffle=False, window_length=window_length,
                                                            window_random=window_random, with_tags=with_tags, with_tids=with_tids,
                                                            merge_tags=merge_tags, num_tags=num_tags)[-1]

    ROC_AUC = tf.keras.metrics.AUC(curve='ROC', name='ROC_AUC',  dtype=tf.float32)
    PR_AUC = tf.keras.metrics.AUC(curve='PR', name='PR_AUC', dtype=tf.float32)

    for entry in dataset:

        audio_batch, label_batch = entry[0], entry[1]

        logits = model(audio_batch, training=False)

        ROC_AUC.update_state(label_batch, logits)
        PR_AUC.update_state(label_batch, logits)

    print('ROC_AUC: ', np.round(ROC_AUC.result().numpy()*100, 2), '; PR_AUC: ', np.round(PR_AUC.result().numpy()*100, 2))

def predict(model, audio, config, threshold=0.5, db_path='/srv/data/urop/clean_lastfm.db'):
    ''' Predicts tags given audio for one track 
    
    Paramters:
    ---------
    model : tf.keras.Model object
        model to use for prediction

    audio : list or array-like

    audio_format : {'waveform', 'log-mel-spectrogram'}
        audio format used in model

    with_tags : list
        list of tags used during training 

    sample_rate : int
        sample rate used for the audio

    threshold : float
        threshold for confidence value of tags to be returned

    db_path : str
        path to lastfm database

    Returns:
    -------
    tags : array
        contains pairs [tag, val] of predicted tags along with the confidence value
    
    '''

    fm = lastfm.LastFm(db_path)

    # compute average by using a moving window
    logits = tf.reduce_mean(model(audio, training=False), axis=[0])
    
    # get tags
    tags = []
    for idx, val in enumerate(logits):
        if val >= threshold:
            tags.append([float(val.numpy()), fm.tag_num_to_tag(int(with_tags[idx]))])
            
    tags = sorted(tags, key=lambda x:x[0], reverse=True)
    return tags

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
            
            
            
