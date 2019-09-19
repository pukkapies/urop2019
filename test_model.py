import os
import argparse
import time

import tensorflow as tf
import numpy as np
import librosa
import audioread

import code.lastfm as q_fm
import projectname_input
import projectname as Model

from code.projectname_train import parse_config

def load_from_checkpoint(audio_format, config, checkpoint_path=None):
    ''' Loads model from checkpoint 
    
    Parameters
    ----------

    audio_format : {'waveform', 'log-mel-spectrogram'}
        audio format of model

    config : argparse.Namespace()
        config generated by parse_config

    checkpoint_path: str or None
        Path to the specific checkpoint to be used. If None will use latest
        checkpoint in checkpoint directory from config

    Returns
    -------
    
    Model loaded from specified checkpoint.

    '''

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
        latest = tf.train.latest_checkpoint(config.log_dir)
        print('Loading from {}'.format(latest))
        checkpoint.restore(latest)

    return model

def get_audio(mp3_path, audio_format, config, array=None, array_sr=None):
    ''' Loads and converts a .mp3 file to format used by model. 
    
    Paramters
    ---------
    mp3_path : str
        path to mp3 file

    audio_format : {'waveform', 'log-mel-spectrogram'}
        audio format to use

    config : argparse.Namespace()
        config generated by parse_config
        
    array: numpy.ndarray
        input audio in array form
        
    array_sr: int
        sampling rate of the input numpy array

    Returns
    ------

    array : np.array
        processed audio array, ready for training
    
    '''
    if mp3_path:
        array, sr_in = librosa.core.load(mp3_path, sr=None, mono=False)
    elif array is not None:
        array = array.astype(np.float32)
        sr_in = array_sr
        
    array = librosa.core.to_mono(array)
    array = librosa.resample(array, sr_in, config.sample_rate)

    if audio_format == "log-mel-spectrogram":
        array = librosa.core.power_to_db(librosa.feature.melspectrogram(array, config.sample_rate, n_mels=config.n_mels))
        array = array.astype(np.float32)
        # normalization
        mean, variance = tf.nn.moments(tf.constant(array), axes=[0,1], keepdims=True)
        array = tf.nn.batch_normalization(array, mean, variance, offset = 0, scale = 1, variance_epsilon = .000001).numpy()

    return array

def test(model, tfrecords_dir, audio_format, split, batch_size=64, window_size=15, merge_tags=None, random=False, with_tags=None, with_tids=None):
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

    window_size : int
        size in seconds of window to be extracted from each audio array
        
    merge_tags : list of list of int
        e.g. [[1,2], [2,3]] means merge 1 with 2, and 2 with 3 respectively

    random : bool
        Specifies if windows should be extracted from a random location, or from the center of the array

    with_tags : list
        list of tags used during training 

    with_tids : list
        list of tids used during training
        
    '''

    # loading test dataset
    dataset = projectname_input.generate_datasets_from_dir(tfrecords_dir, audio_format, split=split, 
                                                            batch_size=batch_size, shuffle=False, window_size=window_size,
                                                            random=random, with_tags=with_tags, with_tids=with_tids,
                                                            merge_tags=merge_tags, num_tags=155, num_epochs=1)[-1]

    ROC_AUC = tf.keras.metrics.AUC(curve='ROC', name='ROC_AUC',  dtype=tf.float32)
    PR_AUC = tf.keras.metrics.AUC(curve='PR', name='PR_AUC', dtype=tf.float32)

    for entry in dataset:

        audio_batch, label_batch = entry[0], entry[1]

        logits = tf.multiply(model(audio_batch, training=False), tf.constant(0.001, dtype=tf.float32))

        ROC_AUC.update_state(label_batch, logits)
        PR_AUC.update_state(label_batch, logits)

    print('ROC_AUC: ', np.round(ROC_AUC.result().numpy(), 2), '; PR_AUC: ', np.round(PR_AUC.result().numpy(), 2))

def get_slices(audio, audio_format, sample_rate, window_size=15):
    ''' Extracts slice of audio along an entire audio array
    
    Parameters
    ----------

    audio : list or array-like
    
    audio_format : {'waveform', 'log-mel-spectrogram'}
        audio format used in model

    sample_rate : int
        sample rate used for the audio

    window_size : int
        size of window to be extracted

    Returns
    -------
    np.array of slices extracted
    '''
    
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

    cutoff : float
        threshold for confidence value of tags to be returned

    window_size : int
        size of windows to be extracted along the audio

    db_path : str
        path to lastfm database

    Returns:
    -------
    tags : array
        contains pairs [tag, val] of predicted tags along with the confidence value
    
    '''

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
            
    tags = sorted(tags, key=lambda x:x[0], reverse=True)
    return tags

if __name__ == '__main__':

    parser = argparse.ArgumentParser()     
    parser.add_argument("format", help="Model audio format")
    parser.add_argument("mode", choices=["predict", "test"], help="Choose functionality of script, testing or predict")
    parser.add_argument("config", help="Path to config JSON file")
    parser.add_argument("--checkpoint", help="Path to a checkpoints, will default to directory in config.")
    parser.add_argument("--lastfm-path", help="Path to lastfm database", default="/home/calle/clean_lastfm.db")
    parser.add_argument("--tfrecords-dir", help="Path to tfrecords directory, specify if test mode has been selected")
    parser.add_argument("--mp3-path", help="Path to mp3 dir or mp3 file to predict")
    parser.add_argument("--from-recording", help="If True, the input audio will be recorded from your microphone", action="store_true")
    parser.add_argument("-s", "--recording-second", help="Number of seconds to record. Minimum length is 15 seconds", type=int, default='15')
    parser.add_argument("--cutoff", type=float, help="Lower bound for what prediction values to print", default=0.1)

    args = parser.parse_args()
    print(args)

    config = parse_config(args.config, args.lastfm_path)[0]
    model = load_from_checkpoint(args.format, config, checkpoint_path=args.checkpoint) 
    print(type(model))

    if args.mode == "test":
        test(model, args.tfrecords_dir, args.format, config.split, batch_size=config.batch_size, random=config.window_random, with_tags=config.tags, merge_tags=config.tags_to_merge)
    else:
        
        if not (args.mp3_path or args.from_recording):
            raise ValueError("If predicting, must either specify mp3 file(s) to predict, or set --from-recording as True")
        elif (args.mp3_path and args.from_recording):
            raise ValueError("If predicting, must either specify mp3 file(s) to predict, or set --from-recording as True")
        elif args.mp3_path:
            if os.path.isfile(args.mp3_path):
                try:
                    audio = get_audio(config=config, mp3_path=args.mp3_path)
                    print("prediction: ", predict(model, config, audio, cutoff=args.cutoff))
                except audioread.NoBackendError:
                    print('skipping {} due to NoBackendError.'.format(args.mp3_path))
                
            else:
                for path in os.listdir(args.mp3_path): 
                    try:
                        audio = get_audio(config=config, mp3_path=os.path.join(args.mp3_path, path))
                    except audioread.NoBackendError:
                        print('skipping {} due to NoBackendError.'.format(path))
                        continue
                
                    print("file: ", path)
                    print("prediction: ", predict(model, config, audio, cutoff=args.cutoff))
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
            audio = get_audio(config=config, array=audio, array_sr=sr_rec)
            print("prediction: ", predict(model, audio, args.format, config.tags, sr_rec, cutoff=args.cutoff, db_path=args.lastfm_path))
            print("prediction: ", predict(model, audio, args.format, config.tags, config.sample_rate, cutoff=args.cutoff, db_path=args.lastfm_path))
            
            
            
