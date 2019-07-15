'''

'''

import os

import librosa
import numpy as np
import pandas as pd


root_dir = '/srv/data/urop/7digital_numpy/'

if __name__ = '__main__':
    for i in range(10):
        for j in range(10):
            dir = os.path.join()
            for file_name in os.listdir(dir):
                path = os.path.join(dir, file_name)

                file = np.load(path)

                n_channels = file['array'].shape[0]
                for i in range(n_channels):
                    file['array'][i] = librosa.feature.melspectrogram(
                                        file['array'][i], int(file['sr']))

                # Turning it into a log-mel spectogram
                file['array'] = np.log(file['array'])

                # TODO: Output this to where????
