''' Script for converting waveforms into spectrograms and saving as TFRecords file

'''

# TODO: Maybe "recycle" this script to also create TFRecords from raw waveforms.
# Could utilize arguments for this

import os

import librosa
import numpy as np
import tensorflow as tf

from .modules import query_lastfm as q_fm
from .modules import query_msd_summary as q_msd


root_dir = '/srv/data/urop/7digital_numpy/'
TAGS = [] # Allowed tags

def encode_tags(tags):
    ''' Encodes tags for the TFRecords file '''
    # TODO
    return

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    with tf.python_io.TFRecordWriter(tf_filename) as writer: # TODO: Decide filename 
        # Looping through all of the directories containing .npz files
        for i in range(10):
            for j in range(10):

                dir = os.path.join(root_dir, str(i), str(j)) # Current directory
                # Loop through files in directory
                for file_name in os.listdir(dir):

                    path = os.path.join(dir, file_name)
                    # TODO: Get TID using Davides+Adens new database
                    # id_7digital = file_name[:-4]
                    tid = ""
                                    
                    file = np.load(path)
                    sr = int(file['sr']) 

                    # Converting to mono if 2 channels (all files either 1 or 2)
                    if file['array'].shape[0] == 2:
                        array_mono = librosa.core.to_mono(file['array'])
                    else:
                        array_mono = file['array']
                   
                    # TODO: "Up-sample/down-sample?"

                    # Getting log-mel-spectrogram
                    # Could probably do have a if arg == "spectrogram" here (maybe also MFCC?)
                    spectrogram = np.log(librosa.feature.melspectrogram(array_mono, sr))

                    tags = q_fm.get_tags(tid) 
                    
                    # TODO: Encode+filter tags (after friday when we decide how)
                    # filter will probably be something like:
                    # tags = [tag for tag in tags if tag in TAGS]
                    # We just need a function to get the TAGS.

                    # TODO: Refine following outline of the saving to TFRecords procedure
                    spectrogram_str = tf.io.serialize_tensor(tf.convert_to_tensor(spectrogram))
                    example = tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    'spectrogram' : _bytes_feature(spectrogram_str)
                                    'tid' :         _bytes_feature(bytes(tid)) 
                                    'tags' :        # TODO: After knowing encoding?
                            }))

                    writer.write(example.SerializeToString())
