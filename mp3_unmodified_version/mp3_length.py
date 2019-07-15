# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:01:01 2019

@author: MacBook Pro
"""

'''
NOTE: In here, I used 999999999999 in the lengths and sizes
columns to indicate faulty mp3 tracks
'''

import os
import pandas as pd
import numpy as np
from mutagen.mp3 import MP3

MP3_ROOT_DIR = '/srv/data/msd/7digital/'

def mp3_length(ult_path='~/urop2019/ultimate_csv.csv', 
               output_path='~/urop2019/ultimate_csv_size.csv'):

    df = pd.read_csv(ult_path, header=None)
    df.rename(columns={0:'track_id', 1:'track_7digitalid', 2:'path'}, inplace=True)
    df = df.sort_values(by='path')

    size = np.zeros(len(df))
    length = np.zeros(len(df))

    paths = df.path.tolist()

    for i, path in enumerate(paths):
        _ = MP3_ROOT_DIR[:-1] + path
    
    #getting file size of mp3
        try:
            size[i] = os.path.getsize(_)
        except:
            size[i] = 999999999999
        
        try:
            audio = MP3(_)
            length[i] = audio.info.length
        except:
            length[i] = 999999999999
        
    
    
    #if i%10000 ==0:
        print(i)
    
    df.loc[:,'sizes']=size
    df.loc[:,'lengths']=length
    df.to_csv(output_path, index=False)
