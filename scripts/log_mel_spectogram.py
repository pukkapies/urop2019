'''

'''

import os

import librosa
import numpy as np
import pandas as pd


root_dir = '/srv/data/urop/7digital_numpy/'

if __name__ = '__main__':
    # Looping through all of the directories containing .npz files
    for i in range(10):
        for j in range(10):
            dir = os.path.join(root_dir, str(i), str(j)) # Current directory
            # Loop through files in directory
            for file_name in os.listdir(dir):
                path = os.path.join(dir, file_name)
                
                file = np.load(path)

                # Getting first spectogram in order to retrieve its shape and initialize
                # the array that will store the spectrograms
                spectrogram = librosa.feature.melspectrogram(
                                    file['array'][0], int(file['sr'])) 
                shape = spectrogram.shape
                array = np.empty(file['array'].shape[0], shape[0], shape[1])
                array[0] = spectrogram

                # Get spectrograms from remaining channels
                n_channels = file['array'].shape[0]
                for i in range(1, n_channels):
                    array[i] = librosa.feature.melspectrogram(
                                        file['array'][i], int(file['sr']))

                # Turning all channels into log-mel spectograms simultaneously
                array = np.log(array)

                # TODO: Output this to where????
