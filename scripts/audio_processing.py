''' Script for processing .npz files and saving as a tfrecords file

Notes
-----
This file can be run as a script, for more information on possible arguments type 
audio_processing -h in the terminal.

IMPORTANT: If using this script elsewhere than on boden then remember to use the option --root-dir
to set directory where the .npz files are stored. The directory needs to have a given layout:
Under the directory, all non-folders must be .npz files. The name of a file is given by the
7digital id and it will be located under "root_dir/digit 1/digit 2/7digital id.npz", where digit 1
and 2 are the first and second digits of the 7digital id

Functions
---------
- process_array             
        Processesing array and applying desired audio format

- get_encoded_tags
        Gets tags for a tid and encodes them with a one-hot vector        

- _bytes_feature
        Creates a BytesList feature

- _float_feature
        Creates a FloatList feature

- _int64_feature
        Creates a Int64List feature

- get_example
        Gets a tf.train.Example object with features containing the array, tid and the encoded tags

-save_examples_to_tffile
        Creates and saves a TFRecord file

-save_split
        Creates and saves 3 TFRecord files for train, validation and test data.
'''

import os
import sys
import time
import argparse

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

if os.path.basename(os.getcwd()) == 'scripts':
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../modules')))
else:
    sys.path.insert(0, os.path.join(os.getcwd(), 'modules'))

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
        unprocessed array, directly from the .npz file

    sr : int
        sample rate

    audio_format : str
        If "log-mel-spectrogram" convert to a log-mel-spectrogram, else
        keep audio as raw waveform.

    Returns
    -------
    ndarray
        processed array
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
        tid

    fm : LastFm object
        LastFm object used to query clean_lastfm.db 

    n_tags : int
        number of entries in clean_lastfm.db

    Returns
    -------
    ndarray
        one-hot vector storing tag information of the tid.
    
    '''
    
    tag_nums = fm.tid_num_to_tag_nums(fm.tid_to_tid_num(tid))

    # returns None if empty, so that it is easy to check for empty tags
    if not tag_nums:
        return
    
    # encodes the tags
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
        ndarray containing audio data.

    tid : str

    encoded_tags : ndarray
        ndarray containing the encoded tags as a one-hot vector 
    
    Returns
    -------
    A tf.train.Example object containing array, tid and encoded_tags as features.
    '''

    example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'spectrogram' : _float_feature(array.flatten()),
                    'tid' :         _bytes_feature(bytes(tid, 'utf8')),
                    'tags' :        _int64_feature(encoded_tags)
            }))
    return example





def save_examples_to_tffile(df, output_path, audio_format, root_dir, tag_path, verbose):
    """ Creates and saves a TFRecord file.

    TODO: More documentation here

    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame containing columns: "trackid" and "file_path"

    output_path : str
        Path or name to save TFRecord file as. If not a path it will save it in the current folder, 
        with output_path as name.

    audio_format : str 
        If "log-mel-spectrogram" audio will be converted to that format, else it will default to raw waveform

    root_dir : str
        root directory to where the .npz files are stored

    tag_path : str
       path to the lastfm_clean.db 

    verbose : bool
        If true, output progress during runtime
    """

    with tf.io.TFRecordWriter(output_path) as writer:

        start = time.time()
        fm = q_fm.LastFm(tag_path)
        # this is used to encode the tags, calculated outside the loop for efficiency
        n_tags = len(fm.get_tag_nums())

        for i, cols in df.iterrows():
            
            if verbose and i % 500 == 0:
                end = time.time()
                print("{} tracks saved. Last 500 tracks took {} s".format(i, end-start))
                start = time.time()

            # unpack columns
            tid, file_path = cols
            path = os.path.join(root_dir, file_path[:-9] + '.npz')

            encoded_tags = get_encoded_tags(tid, fm, n_tags)

            # skip tracks which dont have any "clean" tags    
            if encoded_tags.size == 0:
                if verbose:
                    print("{} as no tags. Skipping...".format(tid))
            
            # loading the unsampled file from path of npz file and process it.
            unsampled_file = np.load(path)
            processed_array = process_array(unsampled_file['array'], 
                                            unsampled_file['sr'], audio_format)
            
            example = get_example(processed_array, tid, encoded_tags)
            writer.write(example.SerializeToString())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--format", help="Set output format of audio, defaults to waveform")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--split", help="train/val/test split, supply as TRAIN/VAL/TEST.")
    parser.add_argument("--num-files", default=10, type=int, help="Number of files to split the data into")
    parser.add_argument("--root-dir", default='/srv/data/urop/7digital_numpy/', help="Set absolute path to directory containing the .npz files, defaults to path on boden")
    parser.add_argument("--tag-path", default='/srv/data/urop/clean_lastfm.db', help="Set absolute path to .db file containing the 'clean' tags.")
    parser.add_argument("--csv-path", default='/srv/data/urop/final_ultimate.csv', help="Set absolute path to ultimate csv file")
    parser.add_argument("--output-dir", default='/srv/data/urop/tf/', help="Set absolute path to output directory")
    parser.add_argument("-i", "--interval", help="Sets which interval of files to process. Supply as START/STOP. Use in combination with --num-files")

    args = parser.parse_args()
    
    # set seed in case interval is specified so that all instances will run on separate parts of the data
    if args.interval:
        np.random.seed(1)
    # gets useful columns from ultimate_csv.csv and shuffles the data.
    df = pd.read_csv(args.csv_path, usecols=["track_id", "file_path"], comment="#").Sample(frac=1).reset_index(drop=True)
    
    # create base name, for naming the TFRecord files
    if args.format == "log-mel-spectrogram":
        base_name = os.path.join(args.output_dir, args.format + "_")
    else:
        base_name = os.path.join(args.output_dir, "waveform_")
    
    # save in a TRAIN/VAL/TEST split if specified
    if args.split: 
        # setting up train, val, test from split and ensuring their sum is 1.
        values = [float(_) for _ in args.split.split("/") ]
        tot = sum(values)
        train, val, test = [val/tot for val in values]


        # splits the DataFrame according to train/val/test.
        size = len(df)
        train_df = df[:size*train]
        test_df = df[size*train:size*(train+val)]
        val_df = df[size*(train+val):]

        # creating + saving the 3 TFRecord files
        name = base_name + args.split + ".tfrecord"
        save_examples_to_tffile(train_df, os.path.join(args.output_dir,"train_"+name), args.format, args.root_dir, args.tag_path, args.verbose)
        save_examples_to_tffile(test_df, os.path.join(args.output_dir, "test_"+name), args.format, args.root_dir, args.tag_path, args.verbose)
        save_examples_to_tffile(val_df, os.path.join(args.output_dir, "val_"+name), args.format, args.root_dir, args.tag_path, args.verbose)

    # otherwise save in args.num_files equal-sized files.
    else:
        # if interval is specified only create files over the given interval.
        if args.interval:
            # getting start and end of interval
            start, stop = [int(_) for _ in args.interval.split("/")]
            
            # if stop is contains the last file this will need to be dealt with separately, as last file will contain
            # the rounding errors, i.e. it will have a size thats slightly bigger than the others.
            if stop >= args.num_files:
                stop = args.num_files-1
                name = base_name + str(args.num_files) + ".tfrecord"
                print("Now writing to: " + name)
                df_slice = df.loc[(args.num_files-1)*len(df)//args.num_files:]
                save_examples_to_tffile(df_slice, name, args.format, args.root_dir, args.tag_path, args.verbose)

            # create and save the files.
            for i in range(start-1, stop):
                name = base_name + str(i+1) + ".tfrecord"
                print("Now writing to: " + name)
                df_slice = df[i*len(df)//args.num_files:(i+1)*len(df)//args.num_files]
                save_examples_to_tffile(df_slice, name, args.format, args.root_dir, args.tag_path, args.verbose)
        else:
            # create and save the num_files files
            for i in range(args.num_files-1):
                name = base_name + str(i+1) + ".tfrecord"
                print("Now writing to: " + name)
                df_slice = df[i*len(df)//args.num_files:(i+1)*len(df)//args.num_files]
                save_examples_to_tffile(df_slice, name, args.format, args.root_dir, args.tag_path, args.verbose)
            name = base_name + str(args.num_files)
            print("Now writing to: " + name) + ".tfrecord"
            df_slice = df.loc[(args.num_files-1)*len(df)//args.num_files:]
            save_examples_to_tffile(df_slice, name, args.format, args.root_dir, args.tag_path, args.verbose)
