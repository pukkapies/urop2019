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
import numpy as np
import os
import pandas as pd
import time

mp3_root_dir = '/srv/data/msd/7digital/'
npz_root_dir = '/srv/data/urop/7digital_numpy/'

if 'path_ult' not in globals():
    path_ult = '/srv/data/urop'

def create_folder_structure():
    '''
    Generate folder structure to store npz files.
    '''
    for dirpath, dirnames, filenames in os.walk(mp3_root_dir):
        structure = os.path.join(npz_root_dir, dirpath[len(mp3_root_dir):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            raise OSError("Directory " + structure + " already exits!!")

def no_sound(df):
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
    
    start = time.time()

    for idx, path in enumerate(df['path'], verbose=False):
        path_npz = npz_root_dir[:-1] + path[:-9]
        start_time = time.time()
        if os.path.isfile(path_np+'.npz'):
            print('Already Exist')
        
        else:
            
            _ = mp3_root_dir[:-1] +path
    
            array, sr = librosa.core.load(_, sr=None, mono=False)
    
            array_mono = librosa.core.to_mono(array)
            array_split = librosa.effects.split(array_mono)
    
            np.savez(path_np, array=array, sr=sr, split=array_split)
        
        if verbose == True:
            if idx % 100 == 0:
                print("{:5d} - TIME ELAPSED: {}".format(idx, time.time()-start))
                

def count(df, final_check=False):
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
    
    count = 0
    l = []
     
    for idx, path in enumerate(df['path']):
        path_npz = npz_root_dir[:-1] +path[:-9]
        if os.path.isfile(path_np + '.npz'):
            count += 1
        elif final_check == True:
            l.append(path)
    
    if final_check == True:
        print("{} OUT OF {} CONVERTED.".format(count, len(df)))
        return l
    else:
        print("{} OUT OF {} CONVERTED.".format(count, len(df)))
        
        
def zip_correction(track_7did):
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
    
    track_7did = str(track_7did)

    path = '/' + track_7did[0] + '/' track_7did[1] + '/' + track_7did + '.clip.mp3'
    path_npz = npz_root_dir[:-1] + path[:-9]
    path = os.path.join(mp3_root_dir, path)
    array, sample_rate = librosa.core.load(path, sr=None, mono=False)
    array_split = librosa.effects.split(librosa.core.to_mono(array))
    np.savez(path_np, array=array, sr=sample_rate, split=array_split)



    
