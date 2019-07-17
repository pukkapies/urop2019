''' Script for converting waveforms into spectrograms and saving as TFRecords file

'''

# TODO: Maybe "recycle" this script to also create TFRecords from raw waveforms.
# Could utilize arguments for this

import os
import sys

import librosa
import numpy as np
import tensorflow as tf

from .modules import query_lastfm as q_fm
from .modules import query_msd_summary as q_msd


root_dir = '/srv/data/urop/7digital_numpy/'
TAGS = [] # Allowed tags


def get_filepaths():
    ''' Gets paths to all .npz files and returns them in a list '''

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
    ''' Returns processed array with desired audio format 
    
    Summary
    -------
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
        desired audio format


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
    if audio_format == "log-mel-spectrogram"
        array = np.log(librosa.feature.melspectrogram(array, 16000))
    elif audio_format == "MFCC":
        # TODO: Maybe some MFCCs??
    
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
    '''  '''
    # TODO: Refine following outline of the saving to TFRecords procedure
    array_str = tf.io.serialize_tensor(tf.convert_to_tensor(array))
    example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'spectrogram' : _bytes_feature(array_str),
                    'tid' :         _bytes_feature(bytes(tid)),
                    'tags' :        # TODO: After knowing encoding?
            }))
    return example


if __name__ == '__main__':

    # TODO: Fix sys args, need to get audio_format 

    paths = get_filepaths()
        
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

"These scripts are still fairly untested and should only be used after we sort out the couple of points"

