''' Contains tools to process the .npz files (or the .mp3 files directly) and create the .tfrecord files.


Notes
-----
This file can be run as a script, for more information on possible arguments type 
audio_processing -h in the terminal.

The script can output .tfrecord files in two different ways, depfilename_suffix on arguments:
if --split TRAIN/VAL/TEST is set then 3 .tfrecord files will be created. A train, validation and test file.
TRAIN, VAL, TEST can be either integer or floats and they dictate what proportion of entries will be saved in each file.
Example: python audio_processing --split 0.9/0.05/0.05 will save 90% of entries to the train file and 5% each 
to the remaining ones.

If --num-files n_fileS is set then n_fileS .tfrecord files will be created, each with the same amount of entries. Furthermore,
if --interval START/STOP is specified then running the script will only create the files between START and STOP, where 
START and STOP are integers. This is useful for splitting up the workload between multiple instances to save time.

If using this script elsewhere than on Boden following arguments will need to be set:
    --root-dir to set root directory of where the .npz files are stored
    --tag-path to set path to clean_lastfm.db, the database containing the cleaned tags
    --csv-path to set path to ultimate.csv, the csv file containing tids that will be used and paths to their .mp3 files
    --output-dir to set what directory the .tfrecord files should be saved to


Functions
---------
- process_array             
    Takes a audio array and applies the desired transformations (resample, convert to desired audio format).

- get_encoded_tags
    Gets tags for a tid and encodes them in a one-hot vector.      

- _bytes_feature
    Creates a BytesList feature.

- _float_feature
    Creates a FloatList feature.

- _int64_feature
    Creates a Int64List feature.

- get_example
    Gets a tf.train.Example object with features containing the array, tid and the encoded tags.

- save_to_tfrecord
    Creates and saves a TFRecord file.
'''

import argparse
import os
import sys
import time

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.utils import Progbar

from lastfm import LastFm
from lastfm import LastFm2Pandas

def process_array(array, audio_format, sr_in, sr_out=16000, n_mels=128, hop_length=512):
    ''' Processesing array and applying desired audio format 
    
    The array is processed by the following steps:
    1. Convert to mono (if not mono already);
    2. Resample to desired sample rate;
    3. Convert audio array to desired audio format.

    Parameters
    ----------
    array: ndarray
        The unprocessed array, as obtained using librosa.core.load().
    
    audio_format: {'waveform', 'log-mel-spectrogram'}
        If 'log-mel-spectrogram', audio will be converted to that format; otherwise, it will default to raw waveform.

    sr_in: int
        The sample rate of the original audio.

    sr_out: int
        The sample rate of the output processed audio.

    n_mels: int
        The number of mels in the mel-spectrogram.

    hop_length: int
        The number of samples between successive frames in the mel-spectrogram.

    Returns
    -------
    ndarray
        The processed array.
    '''
    
    # convert to mono
    if len(array.shape) > 1:
        array = librosa.core.to_mono(array)

    # resample
    array = librosa.resample(array, sr_in, sr_out)
    
    if audio_format == "log-mel-spectrogram":
        array = librosa.core.power_to_db(librosa.feature.melspectrogram(array, sr_out, n_mels=n_mels, hop_length=hop_length))
    
    return array

def get_encoded_tags(fm, tid, n_tags):
    ''' Given a tid gets the tags and encodes them with a one-hot encoding 
    
    Parameters
    ----------
    fm: list, LastFm, LastFm2Pandas
        Any (list of) instance(s) of the tags database(s).

    tid: str
        The track tid.

    n_tags: int
        The number of tag entries to encode.

    Returns
    -------
    ndarray
        A one-hot encoded vector storing tag information of the tid (optionally, a multi-dimensional array if multiple databases are passed).
    '''

    if isinstance(fm, (LastFm, LastFm2Pandas)):
        fm = [fm]

    nd_encoded_tags = []

    for fm in fm:
        try:
            tag_nums = fm.tid_num_to_tag_nums(fm.tid_to_tid_num(tid))
        except:
            return None # no tags have been found
    
        # encodes the tags using a one-hot encoding (if n_tags > len(tag_nums), last entries will be zeros)
        encoded_tags = np.zeros(n_tags, dtype=np.int8)

        # encodes the tags in the array
        for tag_num in tag_nums:
            encoded_tags[tag_num-1] = 1

        nd_encoded_tags.append(encoded_tags)
    
    nd_encoded_tags = np.array(nd_encoded_tags)
    
    if nd_encoded_tags.shape[0] == 1:
        return nd_encoded_tags[0] # return a single hot-encoded vector
    else:
        return nd_encoded_tags

def _bytes_feature(value):
    ''' Creates a BytesList Feature. '''

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    ''' Creates a FloatList Feature. '''

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    ''' Creases a Int64List Feature. '''

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def get_example(array, tid, encoded_tags):
    ''' Gets a tf.train.Example object.
    
    Parameters
    ----------
    array: ndarray
        A ndarray containing audio data.

    tid: str
        The track tid.

    encoded_tags: ndarray
        A ndarray containing the encoded tags as a one-hot vector (might be a two-dimensional array, if multiple tags 
        databases are being processed at the same time).
    
    Returns
    -------
    tf.train.Example
        Contains array, tid and encoded_tags as features.
    '''

    feature_dict = {'audio': _float_feature(array.flatten()), 'tid': _bytes_feature(bytes(tid, 'utf8'))} 

    if len(encoded_tags.shape) == 1:
        feature_dict['tags'] = _int64_feature(encoded_tags)
    else:
        for i, encoded_tags in enumerate(encoded_tags):
            feature_dict['tags_' + str(i)] = _int64_feature(encoded_tags)

    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def save_to_tfrecord(df, output_path, audio_format, root_dir, tag_path, sample_rate=16000, n_mels=128, hop_length=512, multitag=False, verbose=False):
    ''' Creates and saves a TFRecord file.

    Parameters
    ----------
    df: DataFrame
        A pandas DataFrame containing the following columns: "track_id", "mp3_path", "npz_path".

    output_path: str
        The path or filename to save TFRecord file as.
        If not a path, the current folder will be used with output_path as filename.

    audio_format: {'waveform', 'log-mel-spectrogram'}
        If 'log-mel-spectrogram', audio will be converted to that format; otherwise, it will default to raw waveform.

    root_dir: str
        The root directory to where the .npz files (or the .mp3 files) are stored.

    tag_path: str
        The path to the lastfm_clean.db database.

    sample_rate: int
        The sample rate to use when serializing the audio.

    n_mels: int
        The number of mels in the mel-spectrogram.

    hop_length: int
        The number of samples between successive frames in the mel-spectrogram.
    
    multitag: list
        If True, encode multiple tags at the same time (provide as list of filenames; feature names will be 'tags-0', 'tags-1' etc.)

    verbose: bool
        If True, print progress.
    '''

    with tf.io.TFRecordWriter(output_path) as writer:
        if not multitag:
            fm = LastFm(tag_path)
            n_tags = len(fm.get_tag_nums())
        else:
            fm = [LastFm(os.path.join(tag_path, path)) for path in multitag]
            n_tags = np.max([len(fm.get_tag_nums()) for fm in fm]) # if n_tags is different for each database, use max and fill the rest with zeros

        # initialize
        exceptions = []

        df.reset_index(drop=True, inplace=True)

        if verbose:
            progbar = Progbar(len(df)) # create an instance of the progress bar

        for i, cols in df.iterrows():
            if verbose:
                progbar.add(1) # update progress bar

            # unpack cols
            tid, path = cols

            # encode tags
            encoded_tags = get_encoded_tags(fm, tid, n_tags)
            
            # skip tracks which dont have any "clean" tags    
            if encoded_tags is None:
                if verbose:
                    print("WARNING skipping {} as no tags has been found for one of the tags db".format(tid))
                continue

            path = os.path.join(root_dir, path)

            if set(df.columns) == {'track_id', 'npz_path'}:
                # get the unsampled array from the .npz file
                unsampled_audio = np.load(path)
            else:
                # get the unsampled array from the original .mp3 file
                try:
                    array, sr = librosa.core.load(path, sr=None)
                except:
                    exceptions.append({'path': path, 'tid': tid, 'encoded_tags': encoded_tags})
                    continue
                unsampled_audio = {'array': array, 'sr': sr}

            # resample audio array into 'sample_rate' and convert into 'audio_format'
            processed_array = process_array(unsampled_audio['array'], audio_format, sr_in=unsampled_audio['sr'], sr_out=sample_rate, n_mels=n_mels, hop_length=hop_length)
            
            # load the tf.Example
            example = get_example(processed_array, tid, encoded_tags)
            
            # save the tf.Example into a .tfrecord file
            writer.write(example.SerializeToString())
    
        # print exceptions
        if set(df.columns) == {'track_id', 'npz_path'}:
            return
        else:
            if exceptions:
                for exception in exceptions:
                    print('WARNING could not process track {} due to unknown audio error ({})'.format(exception["tid"], exception["path"]))
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("format", choices=["waveform", "log-mel-spectrogram"], help="output format of audio")
    parser.add_argument("output", help="directory to save .tfrecord files in")
    parser.add_argument("--root-dir", help="set path to directory containing the .npz or .mp3 files (defaults to path on Boden)", default='/srv/data/msd/7digital/')
    parser.add_argument("--csv-path", help="set path to .csv file (defaults to path on Boden)", default='/srv/data/urop/ultimate.csv')
    parser.add_argument("--tag-path", help="set path to 'clean' tags database (defaults to path on Boden)", default='/srv/data/urop/clean_lastfm.db')
    parser.add_argument("--tag-path-multi", help="set path to multiple 'clean' tags databases", nargs='+')
    parser.add_argument("--mels", help="set number of mel bands in the log-mel-spectrogram, defaults to 128", type=int, default=128)
    parser.add_argument("--hop-length", help="set number of samples between consecutive frames in the log-mel-spectrogram, defaults to 512", type=int, default=512)
    parser.add_argument("--sr", help="set sample rate to use to encode audio, defaults to 16kHz", type=int, default=16000)
    parser.add_argument("-n", "--num-files", help="number of files to split the data into, defaults to 100", type=int, default=100)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-s", "--split", help="percentage of tracks to go in each dataset, supply as TRAIN VAL TEST", type=int, nargs=3)
    mode.add_argument("-i", "--start-stop", help="specify which interval of files to process (inclusive, starts from 1), use in combination with --n-tfrecords, supply as START STOP", type=int, nargs=2)

    args = parser.parse_args()

    # sanity check
    if args.tag_path_multi:
        args.tag_path_multi = [os.path.expanduser(path) for path in args.tag_path_multi]
        args.tag_path = os.path.commonpath(args.tag_path_multi) # tag_paths is interpreted as a root directory when multiple databases are specified
        args.tag_path_multi = [os.path.relpath(path, start=args.tag_path) for path in args.tag_path_multi]
    
    # set seed in order to enable parallel execution
    if args.start_stop:
        np.random.seed(1)

    # get useful columns from ultimate_csv.csv and shuffle the data
    try:
        df = pd.read_csv(args.csv_path, usecols=["track_id", "npz_path"], comment="#").sample(frac=1).reset_index(drop=True)
    except ValueError:
        df = pd.read_csv(args.csv_path, usecols=["track_id", "mp3_path"], comment="#").sample(frac=1).reset_index(drop=True)
    
    # create output folder
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    base_name = os.path.join(args.output, args.format + "_") # to name the output .tfrecord files
    
    # if split is specified, save to three files (for training, validating and testing)
    if args.split: 
        # scaling up split
        tot = len(df)
        split = np.cumsum(args.split) * tot // np.sum(args.split)

        # split the dataframe according to train/val/test split
        df1 = df[:split[0]]
        df2 = df[split[0]:split[1]]
        df3 = df[split[1]:]

        # create and save the three .tfrecord files
        filename = ('train_', 'validation_', 'test+')
        filename_suffix = str(args.split[0]) + '-' + str(args.split[1]) + '-' + str(args.split[2]) + ".tfrecord" 

        for i, df in enumerate((df1, df2, df3)):
            save_to_tfrecord(df, base_name + filename[i] + filename_suffix, audio_format=args.format, 
                            root_dir=args.root_dir, tag_path=args.tag_path, 
                            multitag=args.tag_path_multi,
                            sample_rate=args.sr, n_mels=args.mels, hop_length=args.hop_length,
                            verbose=args.verbose)

    # otherwise, save to args.n_files equal-sized files
    else:
        # if args.start_stop is specified, only create files over the given interval
        if args.start_stop:
            start, stop = args.start_stop
            for n_file in range(start-1, stop):
                filename = base_name + str(n_file+1) + ".tfrecord"
                print()
                print("INFO writing to " + filename)
                # obtain the df slice corresponding to current file
                df_slice = df[n_file*len(df)//args.n_files:(n_file+1)*len(df)//args.n_files]
                # create and save
                save_to_tfrecord(df_slice, filename, audio_format=args.format, 
                                root_dir=args.root_dir, tag_path=args.tag_path, 
                                multitag=args.tag_path_multi,
                                sample_rate=args.sr, n_mels=args.mels, hop_length=args.hop_length,
                                verbose=args.verbose)

            # the last file will need to be dealt with separately, as it will have a slightly bigger size than the others (due to rounding errors)
            if stop >= args.n_files:
                stop = args.n_files-1
                filename = base_name + str(args.n_files) + ".tfrecord"
                print()
                print("INFO writing to " + filename)
                # obtain the df slice corresponding the last file
                df_slice = df.loc[(args.n_files-1)*len(df)//args.n_files:]
                # create and save to the .tfrecord file
                save_to_tfrecord(df_slice, filename, audio_format=args.format, 
                                root_dir=args.root_dir, tag_path=args.tag_path, 
                                multitag=args.tag_path_multi,
                                sample_rate=args.sr, n_mels=args.mels, hop_length=args.hop_length,
                                verbose=args.verbose)

        # otherwise, create all files at once
        else:
            for n_file in range(args.n_files - 1):
                filename = base_name + str(n_file+1) + ".tfrecord"
                print()
                print("INFO writing to " + filename)
                # obtain the df slice corresponding to current file
                df_slice = df[n_file*len(df)//args.n_files:(n_file+1)*len(df)//args.n_files]
                # create and save to the .tfrecord file
                save_to_tfrecord(df_slice, filename, audio_format=args.format, 
                                root_dir=args.root_dir, tag_path=args.tag_path, 
                                multitag=args.tag_path_multi,
                                sample_rate=args.sr, n_mels=args.mels, hop_length=args.hop_length,
                                verbose=args.verbose)

            # the last file will need to be dealt with separately, as it will have a slightly bigger size than the others (due to rounding errors)
            filename = base_name + str(args.n_files) + ".tfrecord"
            print()
            print("INFO writing to " + filename)
            # obtain the df slice corresponding to the last file
            df_slice = df.loc[(args.n_files-1)*len(df)//args.n_files:]
            # create and save to the .tfrecord file
            save_to_tfrecord(df_slice, filename, audio_format=args.format, 
                                root_dir=args.root_dir, tag_path=args.tag_path, 
                                multitag=args.tag_path_multi,
                                sample_rate=args.sr, n_mels=args.mels, hop_length=args.hop_length,
                                verbose=args.verbose)
