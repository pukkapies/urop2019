# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:21:48 2019

@author: MacBook Pro
"""

'''
Notes
-----
This module will create tools that can be used to analyse different features of
the mp3 tracks by get_faulty_mp3.py ---length, file size, sections that are 
silent. The procedure can be summarised into three categories:
    
-Preparation of dataset: Prepare a dataset to identify broken mp3 files by 
mp3_length(). Then filter the tracks so that all track has at least on tag 
using pre_no_sound().

-Saving information on silent details: no_sound() saves the information of the 
silent details as npz files, and also the converted arrays and sampling
rate of the original MP3 to speed up the loading process in the future.

-Progress check: count() can be used to check progress of no_sound() on
seperate window.

Examples
--------
    create_folder_structure()
    
    mp3_length()
    
    no_sound(start=0, end=40000)
    
    On a seperate window: count()
    


Functions
---------
- set_path_ult
    Tell the script the path of where 'ultimate_csv.csv' was stored.

- create_folder_structure  
    This copies the folder structure of how the mp3 files are saved and produce 
    the same folder structure under a new directory, which will be used 
    to save the converted numpy arrays later.
    
- mp3_length
    Extend the columns of ultimate_csv to identify tracks and return the 
    lengths and sizes of tracks
                           
- pre_no_sound 
    Prepare a dataframe that will be used by no_sound.  
                           
- no_sound                 
    Save three numpy arrays per track:
    1. loaded mp3 arrays    2. sampling rate 
    3. start and end positoin of arrays in 1 when volume of track is above 
    60dB (non-silent). See the librosa documentation on librosa.effect.split 
    for more details. 
                           
                              
- count                    
    Return the number of mp3 files that have been saved as npz files. Return 
    the path of tracks that have not been converted yet if final_check mode 
    is enabled.
                           
- zip_correction           
    Searches for zip files errors and is an error handling tool used in 
    get_faulty_mp3.py


'''

import librosa
import pandas as pd
import time
import numpy as np
import os
import get_faulty_mp3
from mutagen.mp3 import MP3

MP3_ROOT_DIR = '/srv/data/msd/7digital/'
NUMPY_ROOT_DIR = '/srv/data/urop/7digital_numpy/'
if 'path_ult' not in globals():
    path_ult = '/srv/data/urop'

def set_path_ult_no_sound(new_path):
    '''
    Parameters
    ----------
    
    
    new_path: str
        The path where 'ultimate_csv.csv' is stored.
        
    '''
    global path_ult
    path_ult = new_path

def create_folder_structure():
    
    '''
    Returns
    -------
    
    New folder structure in the NUMPY_ROOT_DIR
    '''
    for dirpath, dirnames, filenames in os.walk(MP3_ROOT_DIR):
        structure = os.path.join(NUMPY_ROOT_DIR, dirpath[len(MP3_ROOT_DIR):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder does already exits!")

def mp3_length(ult_file='ultimate_csv.csv', 
               output_file='ultimate_csv_size.csv'):
    
    '''
    Parameters
    ----------
    
    ult_file: The file name of the input csv. Default- ultimate_csv.csv
    output_file: The file name of the output csv. Default-ultimate_csv_size.csv.
    
    Returns
    -------
    csv: A csv that has two extra columns:
        'sizes': float
            The file size of mp3 tracks.
            
        'lengths': float
            The duration of the mp3 tracks.
            
        NOTE: 999999999999 is returned to the corresponding rows for lengths 
        and sizes if the script cannot read the size of the tracks or cannot 
        open the tracks (i.e. broken tracks).
        
    
    '''

    ult_path = os.path.join(path_ult, ult_file)
    output_path = os.path.join(path_ult, output_file)

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

def no_sound(start, end, file='ultimate_csv_size.csv'):
    
    '''
    Parameters
    ----------
    
        start: int
            The index of starting point in the pre_no_sound.csv.
            
        end: int
            The index of ending point in the pre_no_sound.csv.
            
        file: str
            File name of input csv. Default-Ultimate_csv_size.csv
         
    
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
 

    df = pre_no_sound(file)

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
                

def count(final_check=False, file='ultimate_csv_size.csv'):
    
    '''
    Parameters
    ----------
    final_check: bool
        final check mode.
        
    file: str
        File name of input csv. Default-Ultimate_csv_size.csv.

        
    Returns
    -------
    LIST: list
        If it is in final check mode, it returns the list of path of the tracks
        whose mp3 has not been loaded and saved to numpy array
        
    Counter: int
        The number of mp3s that have been loaded and saved as numpy array.
        
    
    '''
    
    df = pre_no_sound(file)
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
    
    Parameters
    ----------
    track_7digitalid: int
        The track_7digitalid of the track.
        
        
    Returns
    -------
    npz file:
        The npz file described in the function no_sound()
    
    '''
    
    
    path = '/'+str(track_7digitalid)[0]+'/'+str(track_7digitalid)[1]+'/'+str(track_7digitalid)+'.clip.mp3'
    path_np = NUMPY_ROOT_DIR[:-1] +path[:-9]
    _ = MP3_ROOT_DIR[:-1] +path
    array, sr = librosa.core.load(_, sr=None, mono=False)
    array_mono = librosa.core.to_mono(array)
    array_split = librosa.effects.split(array_mono)
    np.savez(path_np, array=array, sr=sr, split=array_split)



    
