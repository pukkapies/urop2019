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

# TODO: Maybe "recycle" this script to also create TFRecords from raw waveforms.
# Could utilize arguments for this

import os
import sys
import argparse

import librosa
import numpy as np
import tensorflow as tf
import random

if os.path.basename(os.getcwd()) == 'scripts':
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../modules')))
else:
    sys.path.insert(0, os.path.join(os.getcwd(), 'modules'))

import query_lastfm as q_fm
import query_msd_summary as q_msd


root_dir = '/srv/data/urop/7digital_numpy/'
TAGS = [] # Allowed tags


def get_filepaths():
    ''' Gets paths to all .npz files 
    
    Returns
    -------
    list
        list containing paths (str) to every file under the root_dir directory.
    
    Notes
    -----
    All of the .npz files are located in the root_dir directory by the following structure:
    The name of each file is given by its 7digital_id, it is then located under the path
    "root_dir/digit 1/digit 2/7digital id.npz". Digit 1 and 2 refers to the first and second
    digit in the 7digital id.
    '''

    paths = [] 
    for i in range(10):
        for j in range(10):
            dir = os.path.join(root_dir, str(i), str(j)) # Current directory
            # Loop through files in directory
            for file_name in os.listdir(dir):
                paths.append(os.path.join(dir, file_name))
    return paths


def process_array(array, sr, audio_format):
    # TODO: Change name of audio format
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

def get_tid_from_path(path):
    '''  '''

    # TODO: Get TID using Davides+Adens new database
    # id_7digital = file_name[:-4]
    return tid

def filter_tags(tags):
    '''  '''

    # TODO
    return tags

def encode_tags(tags):
    ''' Encodes tags for the TFRecords file '''

    # TODO
    return #???

def _bytes_feature(value):
    '''  '''

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
                    'tags' :        encoded_tags # TODO: After knowing encoding?
            }))
    return example





def save_examples_to_tffile(paths, tf_filename, audio_format):

    """
    Given paths to the unsampled files, this function saves the examples(processed array, tid and tags) to a tf_record file
    
    Inupts:
    1) paths - list of paths to unsampled tracks
    2) tf_filename - name of TFRecord file you want to save to
    3) audio_format - format we want to save in check process array for more details
    """

    with tf.python_io.TFRecordWriter(tf_filename) as writer: # TODO: Decide filename 
        for path in paths:

            # Loading the unsampled file from path of npz file   
            unsampled_file = np.load(path)
            processed_array = process_array(unsampled_file['array'], 
                                            unsampled_file['sr'], audio_format)

            # TODO: make get_tid_from_path()
            tid = get_tid_from_path(path)
            tags = q_fm.get_tags(tid) 
            # TODO: make filter_tags() (when we decide how)
            tags = filter_tags(tags)
            # TODO: make encode_tags() (when we decide how)
            encoded_tags = encode_tags(tags) 

            # TODO: Refine get_example() 
            example = get_example(processed_array, tid, encoded_tags)
            writer.write(example.SerializeToString())




if __name__ == '__main__':

    # TODO: Maybe add more arguments?? train/val/test maybe?
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--format", help="Set output format of audio, defaults to waveform")
    parser.add_argument("-s", "--split", help"train/val/test split, supply as TRAIN/VAL/TEST")
    parser.add_argument("--root-dir", help="Set absolute path to directory containing the .npz files, defaults to path on boden")
    
    args = parser.parse_args()

    train = 0.7
    val = 0.2
    test = 0.1

    if args.root_dir:
       root_dir = args.path 
    if args.split 
        values = [float(_) for _ in args.split.split("/") ]
        tot = sum(values)
        train, val, test = [val/tot for val in values]
    
    paths = get_filepaths()
    np.random.seed(1)
    paths = np.random.shuffle(paths)
    size = len(paths)

    train_paths = paths[:size*train]
    test_paths = paths[size*train:size*(train+val)]
    val_paths = paths[size*(train+val):]

    save_examples_to_tffile(train_paths, "tf_train_"+args.format, args.format)
    save_examples_to_tffile(test_paths, "tf_test_"+args.format, args.format)
    save_examples_to_tffile(val_paths, "tf_val_"+args.format, args.format)
