# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:21:48 2019

@author: MacBook Pro
"""

'''
Notes
-----


Functions
---------
- create_folder_structure  This copies the folder structure of how the mp3 
                           files are saved and produce the same folder 
                           structure under a new directory, which will be used 
                           to save the converted numpy arrays later.
                           
- no_sound                 Save three numpy arrays per track:
                           1. loaded mp3 arrays    2. sampling rate 
                           3. start and end positoin of arrays in 1 when volume
                              of track is above 60dB (non-silent)
                           See the librosa documentation on 
                           librosa.effect.split for more details.
                           
                              
- count                    Return the number of mp3 files that have been saved
                           as npz files. Return the path of tracks that have
                           not been converted yet if final_check mode is 
                           enabled.
                           
- zip_correction           Searches for zip files errors and is an error 
                           handling tool used in get_faulty_mp3.py


'''


import librosa
import pandas as pd
import time
import numpy as np
import os

MP3_ROOT_DIR = '/srv/data/msd/7digital/'
NUMPY_ROOT_DIR = '/srv/data/urop/7digital_numpy/'

def create_folder_structure():
    for dirpath, dirnames, filenames in os.walk(MP3_ROOT_DIR):
        structure = os.path.join(NUMPY_ROOT_DIR, dirpath[len(MP3_ROOT_DIR):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder does already exits!")



#ult_path = 'D://UROP/millionsongsubset_full/MillionSongSubset/ultimate_csv.csv'
def no_sound(start, end, ult_path='~/urop2019/pre_no_sound.csv'):
    
    '''
    Parameters
    ----------
    
        start: int
            The index of starting point in the pre_no_sound.csv.
            
        end: int
            The index of ending point in the pre_no_sound.csv.
            
        ult_path: str
            The directory of the csv that will be using to create the npz file.
            Default--pre_no_sound.csv
         
    
    Returns
    -------
    
    npz files:
        The npz file is of the form ['array', 'sr', 'split']
        'array': 
            The loaded mp3 files in numpy-array form by library -- librosa.
            The array represent the mp3s in its original sampling rate, and 
            multi-channel is preserved. (each row represents one channel)
            
        'sr':
            The sampling rate of the mp3 file.
            
        'split':
            All the sections of the track which is non-silent (>=60dB). The
            information is saved in the form: n*2 numpy.ndarray, and each row
            represent one section -- starting position and ending position of 
            array respectively.
    
    '''
 

    df = pd.read_csv(ult_path)

    paths = df.path.tolist()[start:end]



    for i, path in enumerate(paths):
        path_np = NUMPY_ROOT_DIR[:-1] +path[:-9]
        start_time = time.time()
        if os.path.isfile(path_np+'.npz'):
            print('Already Exist')
        
        else:
            
            _ = MP3_ROOT_DIR[:-1] +path
    
            array, sr = librosa.core.load(_, sr=None, mono=False)
    
            array_mono = librosa.core.to_mono(array)
            array_split = librosa.effects.split(array_mono)
    
            np.savez(path_np, array=array, sr=sr, split=array_split)
        
        if i%100==0:
            print(time.time()-start_time)
            print(i)
                

def count(final_check=False, ult_path='~/urop2019/pre_no_sound.csv'):
    
    '''
    Parameters
    ----------
    final_check: bool
        final check mode.
        
    ult_path: str
        The directory of the csv that was used to create the npz file.
        Default--pre_no_sound.csv
        
    Returns
    -------
    LIST: list
        If it is in final check mode, it returns the list of path of the tracks
        whose mp3 has not been loaded and saved to numpy array
        
    Counter: int
        The number of mp3s that have been loaded and saved as numpy array.
        
    
    '''
    
    df = pd.read_csv(ult_path)
    paths = df.path.tolist()
    counter = 0
    LIST=[]
     
    for i, path in enumerate(paths):
        path_np = NUMPY_ROOT_DIR[:-1] +path[:-9]
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
        
        
def zip_correction(track_7digitalid):
    '''
    NEED CHANGES IN PATHS?
    
    Parameters
    ----------
    track_7digitalid: int
        The track_7digitalid of the track.
        
    Returns
    -------
    npz file:
        The npz file described in the function no_sound()
    
    '''
    
    
    path = '/'+str(track_7digitalid)[0]+'/'+str(track_7digitalid)[1]+'/'
    +str(track_7digitalid)+'.clip.mp3'
    path_np = NUMPY_ROOT_DIR[:-1] +path[:-9]
    _ = MP3_ROOT_DIR[:-1] +path
    array, sr = librosa.core.load(_, sr=None, mono=False)
    array_mono = librosa.core.to_mono(array)
    array_split = librosa.effects.split(array_mono)
    np.savez(path_np, array=array, sr=sr, split=array_split)



    
