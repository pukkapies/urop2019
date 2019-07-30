''' Script for processing .npz files and saving as a TFRecords file

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
- get_filepaths             Gets paths to all .npz files contained under root_dir.
- process_array             Process array and apply desired audio format.
- get_tid_from_path         Gets tid associated to file, given path.
- filter_tags               TODO
- encode_tags               TODO
- _bytes_feature            Creates a BytesList feature.
- get_example               Gets a tf.train.Example object given array, tid and encoded_tags.

'''

import os
import sys
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
    audio_format : {"log-mel-spectrogram", "MFCC", "waveform"}
        desired audio format, if none of the above it defaults to "waveform"


    Returns
    -------
    ndarray
        processed array
    '''
    
    # Converting to mono
    if array.shape[0] == 2:
        array = librosa.core.to_mono(array)
    else:
        array = unprocessed_array
   
    # Resampling the file to 16kHz 
    array = librosa.resample(array, sr, 16000)
    
    # Something along these lines?? Very likely to be changed given
    # how we choose to incorporate sysarg
    if audio_format == "log-mel-spectrogram":
        array = np.log(librosa.feature.melspectrogram(array, 16000))
    elif audio_format == "MFCC":
        # TODO: Maybe some MFCCs??
        array = "???"
    
    return array

def get_encoded_tags(tid):
    ''' Given a tid gets the tags and encodes them with a one-hot encoding '''
    
    lastfm = q_fm.LastFm(PATH)
    tag_nums = lastfm.tid_num_to_tag_num(lastfm.tid_to_tid_num(tid)).sort()

    encoded_tags = ""
    for num in tag_nums:
        encoded_tags += ((num-1)-len(encoded_tags))*"0" + "1"

    return encoded_tags


def _bytes_feature(value):
    ''' Creates a BytesList Feature '''

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_example(array, tid, encoded_tags):
    ''' Gets a tf.train.Example object
    
    Parameters
    ----------
    array : ndarray
        ndarray containing audio data.

    tid : str

    encoded_tags : ???
        ???

    
    Returns
    -------
    A tf.train.Example object
    '''
    # TODO: Refine following outline of the saving to TFRecords procedure
    array_str = tf.io.serialize_tensor(tf.convert_to_tensor(array))
    example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'spectrogram' : _bytes_feature(array_str),
                    'tid' :         _bytes_feature(bytes(tid)),
                    'tags' :        _bytes_feature(bytes(encoded_tags))
            }))
    return example





def save_examples_to_tffile(df, tf_filename, audio_format, root_dir, verbose):
    """ Given paths to .npz files, this function processes them and then creates and saves them to a tf_record file 

    TODO: More documentation here

    Parameters
    ----------
    paths : list
        list of paths to unsampled tracks
    tf_filename : str
        name of TFRecord file to save to
    audio_format : {"log-mel-spectrogram", "MFCC", "waveform"}
        desired audio format, if none of the above it defaults to "waveform"
    """

    with tf.python_io.TFRecordWriter(tf_filename) as writer:
        start = time.time()
        for i, cols in df.iterrows():
            # unpack columns
            tid, file_path = cols
            path = os.path.join(root_dir, file_path)

            # Loading the unsampled file from path of npz file and process it.
            unsampled_file = np.load(path)
            processed_array = process_array(unsampled_file['array'], 
                                            unsampled_file['sr'], audio_format)

            encoded_tags = get_encoded_tags(tid) 

            # TODO: Refine get_example() 
            example = get_example(processed_array, tid, encoded_tags)
            writer.write(example.SerializeToString())

            if verbose and i % 500 == 0:
                end = time.time()
                print("{}/{} tracks saved. Last 500 tracks took {} s".format(i, len(df), end-start))
                start = time.time()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--format", help="Set output format of audio, defaults to waveform")
    parser.add_argument("-s", "--split", default='0.7/0.2/0.1' help"train/val/test split, supply as TRAIN/VAL/TEST. Defaults to 0.7/0.2/0.1")
    parser.add_argument("--root-dir", default='/srv/data/urop/7digital_numpy/', help="Set absolute path to directory containing the .npz files, defaults to path on boden")
    
    args = parser.parse_args()
    
    # Setting up train, val, test from args.split and ensuring their sum is 1.
    values = [float(_) for _ in args.split.split("/") ]
    tot = sum(values)
    train, val, test = [val/tot for val in values]
    
    # Gets usefule columns from ultimate_csv.csv and shuffles the data.
    df = pd.read_csv(PATH, usecols=["track_id", "file_path"], comment="#").sample(frac=1).reset_index(drop=True)
    
    # Splits the DataFrame according to train/val/test.
    size = len(df)
    train_df = df[:size*train]
    test_df = df[size*train:size*(train+val)]
    val_df = df[size*(train+val):]
    
    base_name = args.format + "_" + args.split 
    save_examples_to_tffile(train_df, "train_"+base_name, args.format, args.root_dir, args.verbose)
    save_examples_to_tffile(test_df, "test_"+base_name, args.format, args.root_dir, args.verbose)
    save_examples_to_tffile(val_df, "val_"+base_name, args.format, args.root_dir, args.verbose)
