# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:21:48 2019

@author: MacBook Pro
"""

import librosa
import pandas as pd
import time
import numpy as np
import os

def no_sound(start, end):

    ult_path = '~/urop2019/pre_no_sound.csv'

#ult_path = 'D://UROP/millionsongsubset_full/MillionSongSubset/ultimate_csv.csv'

    df = pd.read_csv(ult_path)

    paths = df.path.tolist()[start:end]


#librosa.core.load('D://UROP/tracks/4019379.clip.mp3')

    for i, path in enumerate(paths):
        path_np = '/srv/data/urop/7digital_numpy'+path[:-9]
        start_time = time.time()
        if os.path.isfile(path_np+'.npz'):
            print('Already Exist')
        
        else:
            
            _ = '/srv/data/msd/7digital'+path
    
            array, sr = librosa.core.load(_, sr=None, mono=False)
    
            array_mono = librosa.core.to_mono(array)
            array_split = librosa.effects.split(array_mono)
    
            np.savez(path_np, array=array, sr=sr, split=array_split)
        
        if i%100==0:
            print(time.time()-start_time)
            print(i)
                

def count(final_check=False):
    ult_path = '~/urop2019/pre_no_sound.csv'
    df = pd.read_csv(ult_path)
    paths = df.path.tolist()
    counter = 0
    LIST=[]
     
    for i, path in enumerate(paths):
        path_np = '/srv/data/urop/7digital_numpy'+path[:-9]
        if os.path.isfile(path_np+'.npz'):
            counter+=1
        else:
            if final_check==True:
                LIST.append(path)
    
    if final_check==True:
        print(counter)
        return(LIST)
    else:
        if counter==len(df):
            print("all_done")
        return(counter)
        
        
def zip_correction(trackid):
    path = '/'+str(trackid)[0]+'/'+str(trackid)[1]+'/'+str(trackid)+'.clip.mp3'
    path_np = '/srv/data/urop/7digital_numpy'+path[:-9]
    _ = '/srv/data/msd/7digital'+path
    array, sr = librosa.core.load(_, sr=None, mono=False)
    array_mono = librosa.core.to_mono(array)
    array_split = librosa.effects.split(array_mono)
    np.savez(path_np, array=array, sr=sr, split=array_split)



    
