''' Contains tools to process the .npz files and create the .tfrecord files.


Notes
-----
This file can be run as a script, for more information on possible arguments type 
audio_processing -h in the terminal.

The script can output .tfrecord files in two different ways, depending on arguments:
if --split TRAIN/VAL/TEST is set then 3 .tfrecord files will be created. A train, validation and test file.
TRAIN, VAL, TEST can be either integer or floats and they dictate what proportion of entries will be saved in each file.
Example: python audio_processing --split 0.9/0.05/0.05 will save 90% of entries to the train file and 5% each 
to the remaining ones.

if --num-files NUM_FILES is set then NUM_FILES .tfrecord files will be created, each with the same amount of entries. Furthermore,
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

- save_example_to_tfrecord
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

from modules.query_lastfm import LastFm
from modules.query_lastfm import LastFm2Pandas

def process_array(array, audio_format, sr_in, sr_out = 16000):
    ''' Processesing array and applying desired audio format 
    
    The array is processed by the following steps:
    1. Convert to mono (if not already);
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
        array = librosa.core.power_to_db(librosa.feature.melspectrogram(array, 16000, n_mels=96))
    
    return array

def get_encoded_tags(tid, fm, n_tags):
    ''' Given a tid gets the tags and encodes them with a one-hot encoding 
    
    Parameters
    ----------
    tid: str

    fm: LastFm, LastFm2Pandas
        Any instance of the tags database.

    n_tags: int
        The number of tag entries in the database.

    Returns
    -------
    ndarray
        A one-hot encoded vector storing tag information of the tid.
    '''
    
    tag_nums = fm.tid_num_to_tag_nums(fm.tid_to_tid_num(tid))

    # returns empty array if it has no clean tags, this makes it easy to check later on
    if not tag_nums:
        return np.array([])
    
    # encodes the tags using a one-hot encoding
    encoded_tags = np.zeros(n_tags, dtype=np.int8)
    for num in tag_nums:
        encoded_tags[num-1] = 1

    return encoded_tags

def _bytes_feature(value):
    ''' Creates a BytesList Feature '''

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    ''' Creates a FloatList Feature '''

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    ''' Creases a IntList Feature '''

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def get_example(array, tid, encoded_tags):
    ''' Gets a tf.train.Example object
    
    Parameters
    ----------
    array: ndarray
        The ndarray containing audio data.

    tid: str

    encoded_tags: ndarray
        The ndarray containing the encoded tags as a one-hot vector.
    
    Returns
    -------
    tf.train.Example
        Contains array, tid and encoded_tags as features.
    '''

    example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'audio': _float_feature(array.flatten()),
                    'tid': _bytes_feature(bytes(tid, 'utf8')),
                    'tags': _int64_feature(encoded_tags)
            }))

    return example

def save_example_to_tfrecord(df, output_path, audio_format, root_dir, tag_path, sample_rate=16000, verbose=True):
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

    verbose: bool
        If True, print progress.
    '''

    with tf.io.TFRecordWriter(output_path) as writer:
        start = time.time()
        start_loop = time.time()

        # tags encoded outside the loop for efficiency
        fm = LastFm(tag_path)
        n_tags = len(fm.get_tag_nums())

        # initialize
        exceptions = []

        df.reset_index(drop=True, inplace=True)

        for i, cols in df.iterrows():
            if verbose and i % 10 == 9:
                print("{:3d} tracks saved. Last 10 tracks took {:6.4f} s".format(i+1, time.time()-start_loop))
                start_loop = time.time()

            # unpack cols
            tid, path = cols

            # encode tags
            encoded_tags = get_encoded_tags(tid, fm, n_tags)
            
            # skip tracks which dont have any "clean" tags    
            if encoded_tags.size == 0:
                if verbose:
                    print("{} has no tags. Skipping...".format(tid))
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
            processed_array = process_array(unsampled_audio['array'], audio_format, unsampled_audio['sr'], sample_rate)
            
            # create and save a tf.Example
            example = get_example(processed_array, tid, encoded_tags)
            writer.write(example.SerializeToString())
    
        print("{} tracks saved in {:10.4f} s".format(i, time.time()-start))

        # try to re-handle exceptions (sometimes it works!!); otherwise, skip
        if set(df.columns) == {'track_id', 'npz_path'}:
            return
        else:
            for exception in exceptions:
                print("Handling exception {}...".format(exception['path']), end=" ", flush=True)
                try:
                    array, sr = librosa.core.load(exception['path'], sr=None)
                except:
                    print("FAIL. Skipping...")
                    continue
                print("SUCCESS!")
                unsampled_audio = {'array': array, 'sr': sr}

                # resample audio array into 'sample_rate' and convert into 'audio_format'
                processed_array = process_array(unsampled_audio['array'], audio_format, unsampled_audio['sr'], sample_rate)
                
                # create and save a tf.Example
                example = get_example(processed_array, exception['tid'], exception['encoded_tags'])
                writer.write(example.SerializeToString())
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("format", choices=["waveform", "log-mel-spectrogram"], help="output format of audio")
    parser.add_argument("output", help="directory to save .tfrecords files in")
    parser.add_argument("--root-dir", help="set path to directory containing the .npz files, defaults to path on Boden", default='/srv/data/msd/7digital/')
    parser.add_argument("--tag-path", help="set path to 'clean' tags database, defaults to path on Boden", default='/srv/data/urop/clean_lastfm.db')
    parser.add_argument("--csv-path", help="set path to .csv file, defaults to path on Boden", default='/srv/data/urop/ultimate.csv')
    parser.add_argument("-r", "--sr", help="set sample rate to use to encode audio, defaults to 16kHz", type=int, default=16000)
    parser.add_argument("-n", "--num-files", help="number of files to split the data into, defaults to 100", type=int, default=100)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-s", "--split", help="percentage of tracks to go in each dataset, supply as TRAIN VAL TEST", type=int, nargs=3)
    mode.add_argument("-i", "--start-stop", help="specify which interval of files to process (inclusive, starts from 1), use in combination with --n-tfrecords, supply as START STOP", type=int, nargs=2)

    args = parser.parse_args()
    
    # set seed in case interval is specified in order to enable parallel execution
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

    # create base name variable, for naming the .tfrecord files
    if args.format == "log-mel-spectrogram":
        base_name = os.path.join(args.output, args.format + "_")
    else:
        base_name = os.path.join(args.output, "waveform_")
    
    # if split is specified, save to three files (for training, testing and validating)
    if args.split: 
        # scaling up split
        tot = len(df)
        split = np.cumsum(args.split) * tot // np.sum(args.split)


        # split the DataFrame according to train/val/test split
        df1 = df[:split[0]]
        df2 = df[split[0]:split[1]]
        df3 = df[split[1]:]

        # create + save the three .tfrecord files
        ending = str(args.split[0]) + '-'+str(args.split[1]) + '-'+str(args.split[2]) + ".tfrecord" 
        save_example_to_tfrecord(df1, base_name + "train_" + ending, args.format, args.root_dir, args.tag_path, args.sr, args.verbose)
        save_example_to_tfrecord(df2, base_name + "test_" + ending, args.format, args.root_dir, args.tag_path, args.sr, args.verbose)
        save_example_to_tfrecord(df3, base_name + "valid_" + ending, args.format, args.root_dir, args.tag_path, args.sr, args.verbose)

    # otherwise, save to args.num_files equal-sized files
    else:
        # if args.start_stop is specified, only create files over the given interval
        if args.start_stop:
            start, stop = args.start_stop
            for num_file in range(start-1, stop):
                name = base_name + str(num_file+1) + ".tfrecord"
                print("Now writing to: " + name)
                # obtain the df slice corresponding to current file
                df_slice = df[num_file*len(df)//args.num_files:(num_file+1)*len(df)//args.num_files]
                # create and save
                save_example_to_tfrecord(df_slice, name, args.format, args.root_dir, args.tag_path, args.sr, args.verbose)

            # the last file will need to be dealt with separately, as it will have a slightly bigger size than the others (due to rounding errors)
            if stop >= args.num_files:
                stop = args.num_files-1
                name = base_name + str(args.num_files) + ".tfrecord"
                print("Now writing to: " + name)
                # obtain the df slice corresponding the last file
                df_slice = df.loc[(args.num_files-1)*len(df)//args.num_files:]
                # create and save to the .tfrecord file
                save_example_to_tfrecord(df_slice, name, args.format, args.root_dir, args.tag_path, args.sr, args.verbose)

        # otherwise, create all files at once
        else:
            for num_file in range(args.num_files - 1):
                name = base_name + str(num_file+1) + ".tfrecord"
                print("Now writing to: " + name)
                # obtain the df slice corresponding to current file
                df_slice = df[num_file*len(df)//args.num_files:(num_file+1)*len(df)//args.num_files]
                # create and save to the .tfrecord file
                save_example_to_tfrecord(df_slice, name, args.format, args.root_dir, args.tag_path, args.sr, args.verbose)
            
            # the last file will need to be dealt with separately, as it will have a slightly bigger size than the others (due to rounding errors)
            name = base_name + str(args.num_files) + ".tfrecord"
            print("Now writing to: " + name)
            # obtain the df slice corresponding to the last file
            df_slice = df.loc[(args.num_files-1)*len(df)//args.num_files:]
            # create and save to the .tfrecord file
            save_example_to_tfrecord(df_slice, name, args.format, args.root_dir, args.tag_path, args.sr, args.verbose)
