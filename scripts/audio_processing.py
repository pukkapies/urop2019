''' Contains tools to process .npz files and create the .tfrecords files.


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
    --csv-path to set path to ultimate.csv, the csv file containing tids that will be used and paths to their mp3 files
    --output-dir to set what directory the .tfrecord files should be saved to

Functions
---------
- process_array             
    Processesing array and applying desired audio format.

- get_encoded_tags
    Gets tags for a tid and encodes them with a one-hot vector.      

- _bytes_feature
    Creates a BytesList feature.

- _float_feature
    Creates a FloatList feature.

- _int64_feature
    Creates a Int64List feature.

- get_example
    Gets a tf.train.Example object with features containing the array, tid and the encoded tags.

-save_examples_to_tffile
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../modules')))

import query_lastfm as q_fm

def process_array(array, sr, audio_format):
    ''' Processesing array and applying desired audio format 
    
    The array is processed by the following steps:
    1. Converted to mono (if not already)
    2. Resampling to 16 kHz
    3. Array is converted to a desired audio format

    Parameters
    ----------
    array : ndarray
        The unprocessed array, from the .npz file.

    sr : int
        The audio sample rate, from the .npz file.

    audio_format : str
        If "log-mel-spectrogram", convert to a log-mel-spectrogram; otherwise, keep audio as raw waveform.

    Returns
    -------
    ndarray
        The processed array.
    '''
    
    # converting to mono
    if array.shape[0] == 2:
        array = librosa.core.to_mono(array)
   
    # resampling the file to 16kHz 
    array = librosa.resample(array, sr, 16000)
    
    if audio_format == "log-mel-spectrogram":
        array = librosa.core.power_to_db(librosa.feature.melspectrogram(array, 16000, n_mels=96))
    
    return array

def get_encoded_tags(tid, fm, n_tags):
    ''' Given a tid gets the tags and encodes them with a one-hot encoding 
    
    Parameters
    ----------
    tid : str

    fm : q_fm.LastFm, q_fm.LastFm2Pandas
        Any instance of the tags database.

    n_tags : int
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
    array : ndarray
        The ndarray containing audio data.

    tid : str

    encoded_tags : ndarray
        The ndarray containing the encoded tags as a one-hot vector.
    
    Returns
    -------
    tf.train.Example
        Contains array, tid and encoded_tags as features.
    '''

    example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'audio' : _float_feature(array.flatten()),
                    'tid' : _bytes_feature(bytes(tid, 'utf8')),
                    'tags' : _int64_feature(encoded_tags)
            }))

    return example

def save_examples_to_tffile(df, output_path, audio_format, root_dir, tag_path, verbose):
    ''' Creates and saves a TFRecord file.

    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame containing columns: "track_id", "mp3_path", "npz_path"

    output_path : str
        The path or filename to save TFRecord file as. If not a path the current folder will be used, with output_path as name.

    audio_format : str 
        If "log-mel-spectrogram", audio will be converted to that format; otherwise, it will default to raw waveform.

    root_dir : str
        The root directory to where the .npz files are stored.

    tag_path : str
       The path to the lastfm_clean.db database.

    verbose : bool
        If True, print progress.
    '''

    with tf.io.TFRecordWriter(output_path) as writer:

        if verbose:
            start = time.time()
        # This is used to encode the tags, it is calculated outside the following loop for efficiency
        fm = q_fm.LastFm(tag_path)
        n_tags = len(fm.get_tag_nums())

        for i, cols in df.iterrows():
            
            if verbose and i % 10 == 0:
                end = time.time()
                print("{} tracks saved. Last 10 tracks took {} s".format(i, end-start))
                start = time.time()

            # unpack columns
            tid, npz_path = cols
            # path to the .npz file
            path = os.path.join(root_dir, npz_path)

            # encode tags
            encoded_tags = get_encoded_tags(tid, fm, n_tags)
            
            # skip tracks which dont have any "clean" tags    
            if encoded_tags.size == 0:
                if verbose:
                    print("{} has no tags. Skipping...".format(tid))
                continue
            
            # load the unsampled file from path of npz file and process it.
            unsampled_file = np.load(path)
            processed_array = process_array(unsampled_file['array'], 
                                            unsampled_file['sr'], audio_format)
            
            # create and save a tf.Example
            example = get_example(processed_array, tid, encoded_tags)
            writer.write(example.SerializeToString())

if __name__ == '__main__':
    

    # setting up arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-f", "--format", help="set output format of audio, defaults to waveform (e.g. 'log-mel-spectrogram')")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--split", help="train/val/test split, supply as TRAIN/VAL/TEST. The proportion of TRAIN, VAL, TEST dictates how many entries is in each file.")
    parser.add_argument("--num-files", default=10, type=int, help="number of equal-sized files to split the data into, defaults to 10")
    parser.add_argument("--root-dir", default='/srv/data/urop/7digital/', help="set absolute path to directory containing the .npz files, defaults to path on boden")
    parser.add_argument("--tag-path", default='/srv/data/urop/clean_lastfm.db', help="set absolute path to .db file containing the 'clean' tags, defaults to path on boden")
    parser.add_argument("--csv-path", default='/srv/data/urop/ultimate.csv', help="set absolute path to csv file, defaults to path on boden")
    parser.add_argument("--output-dir", default='/srv/data/urop/tfrecords/', help="set absolute path to output directory, defaults to path on boden")
    parser.add_argument("-i", "--interval", help="set which interval of files to process, supply as START/STOP (use in combination with --num-files)")

    args = parser.parse_args()
    
    # set seed in case interval is specified so that all instances will run on separate parts of the data
    if args.interval:
        np.random.seed(1)
    # gets useful columns from ultimate_csv.csv and shuffles the data
    df = pd.read_csv(args.csv_path, usecols=["track_id", "npz_path"], comment="#").sample(frac=1).reset_index(drop=True)
    
    # create base name, for naming the TFRecord files
    if args.format == "log-mel-spectrogram":
        base_name = os.path.join(args.output_dir, args.format + "_")
    else:
        base_name = os.path.join(args.output_dir, "waveform_")
    
    # save in a TRAIN/VAL/TEST split if split is specified
    if args.split: 
        # setting up train, val, test from split and scaling them to have sum 1.
        values = [float(_) for _ in args.split.split("/") ]
        tot = sum(values)
        train, val, test = [v/tot for v in values]

        # splits the DataFrame according to train/val/test
        size = len(df)
        print(train)
        train_df = df[:int(size*train)]
        test_df = df[int(size*train):int(size*(train+val))]
        val_df = df[int(size*(train+val)):]

        # creating + saving the 3 tfrecord files
        ending = args.split.replace('/', '-') + ".tfrecord" 
        save_examples_to_tffile(train_df, base_name+"train_"+ending, args.format, args.root_dir, args.tag_path, args.verbose)
        save_examples_to_tffile(test_df, base_name+"test_"+ending, args.format, args.root_dir, args.tag_path, args.verbose)
        save_examples_to_tffile(val_df, base_name+"val_"+ending, args.format, args.root_dir, args.tag_path, args.verbose)

    # save to args.num_files equal-sized files.
    else:
        # if args.interval is specified only create files over the given interval.
        if args.interval:
            # getting start and end of interval
            start, stop = [int(_) for _ in args.interval.split("/")]
            
            # if interval contains the last file this will need to be dealt with separately, as last file will contain
            # the rounding errors, i.e. it will have a size thats slightly bigger than the others.
            if stop >= args.num_files:
                stop = args.num_files-1
                name = base_name + str(args.num_files) + ".tfrecord"
                print("Now writing to: " + name)
                # obtaining the df slice corresponding the last file
                df_slice = df.loc[(args.num_files-1)*len(df)//args.num_files:]
                # creating and saving to the .tfrecord file
                save_examples_to_tffile(df_slice, name, args.format, args.root_dir, args.tag_path, args.verbose)

            # create and save the remaining files.
            for i in range(start-1, stop):
                name = base_name + str(i+1) + ".tfrecord"
                print("Now writing to: " + name)
                # obtaining the df slice corresponding to current file
                df_slice = df[i*len(df)//args.num_files:(i+1)*len(df)//args.num_files]
                # creating and saving to the .tfrecord file
                save_examples_to_tffile(df_slice, name, args.format, args.root_dir, args.tag_path, args.verbose)
        # args.split not specified, so creating all files at once
        else:
            # create and save the num_files files
            for i in range(args.num_files-1):
                name = base_name + str(i+1) + ".tfrecord"
                print("Now writing to: " + name)
                # obtaining the df slice corresponding to current file
                df_slice = df[i*len(df)//args.num_files:(i+1)*len(df)//args.num_files]
                # creating and saving to the .tfrecord file
                save_examples_to_tffile(df_slice, name, args.format, args.root_dir, args.tag_path, args.verbose)
            name = base_name + str(args.num_files)
            print("Now writing to: " + name) + ".tfrecord"
            # obtaining the df slice corresponding to the last file
            df_slice = df.loc[(args.num_files-1)*len(df)//args.num_files:]
            # creating and saving to the .tfrecord file
            save_examples_to_tffile(df_slice, name, args.format, args.root_dir, args.tag_path, args.verbose)
