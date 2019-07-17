'''

'''

import os

import librosa
import numpy as np
from .modules import query_lastfm as q_fm
from .modules import query_msd_summary as q_msd


root_dir = '/srv/data/urop/7digital_numpy/'
TAGS = [] # Allowed tags

def get_tags_7digitalid(id):
    ''' Gets tags using 7digitalid
    
    Parameters
    ----------
    id : str or int 
        The 7digital id.

    Returns
    -------
    list of tags
    '''

    return q_fm.get_tags(q_msd.get_trackid_from_7digitalid(id))


def encode_tags(tags):
    ''' Encodes tags for the TFRecords file '''
    return



if __name__ == '__main__':
    # Looping through all of the directories containing .npz files
    for i in range(10):
        for j in range(10):

            dir = os.path.join(root_dir, str(i), str(j)) # Current directory
            # Loop through files in directory
            for file_name in os.listdir(dir):

                path = os.path.join(dir, file_name)
                id = file_name[:-4]
                                
                file = np.load(path)
                sr = int(file['sr']) 

                # Converting from stereo (2 channels) to mono 
                array_mono = librosa.core.to_mono(file['array'])
               
                # TODO: "Up-sample/down-sample?"

                # Getting log-mel-spectrogram
                spectrogram = np.log(librosa.feature.melspectrogram(array_mono, sr))

                tags = get_tags_7digitalid(id)
                
                # TODO: Encode+filter tags (after friday when we decide how)
                # TODO: Output this to where????



