"""
"""
''' Contains tools for fetching mp3 files on the server, matching 7digitalid's with tid's, and purging unwanted entries such as mismatches, faulty mp3 files, tracks without tags or duplicates


Notes
-----
This file can be run as a script. To do so, just type 'python mp3_to_npz.py' in the terminal. The help 
page should contain all the options you might possibly need. You will first need to run track_fetch.py and
provide the output of that script as an input argument for this one.

This script performs the following operations:
    
- preparing the directory: see create_folder_structure;

- saving information on silence: no_sound() saves the information about silence in npz files, as well
                                 as the converted arrays and sampling rate of the original mp3 to speed up 
                                 the loading process in the future;

- progress check: no_sound_count() can be used to check progress of no_sound() on seperate window.


Functions
---------
- set_mp3_root_dir
    Tell the script the root directory of where mp3s were stored.
    
- set_npz_root_dir
    Tell the script the root directory of where numpy arrays will be stored.

- create_folder_structure  
    Copy the folder structure of how the mp3 files are saved and produce 
    the same folder structure under a new directory, which will be used to save
    the converted numpy arrays.

- mp3_path_to_npz_path
    Convert the path of a mp3 file into the path of the corresponding npz file.

- savez
    Convert a mp3 files into three numpy arrays as an npz file:
    1. loaded mp3 arrays;
    2. sample rate;
    3. start and end positoin of arrays when volume of track is above 60dB (non-silent).
                                 
- no_sound                 
    Apply savez() to the tracks provided provided by the input dataframe.
                                         
- no_sound_count                    
    Return the number of mp3 files that have been correctly saved as npz files. 
    Return the paths of the tracks that have not been saved yet if final_check is True.
'''

import argparse
import os
import time

import librosa
import numpy as np
import pandas as pd

mp3_root_dir = '/srv/data/msd/7digital/'
npz_root_dir = '/srv/data/urop/7digital_numpy/'

def set_mp3_root_dir(new_root_dir):
    ''' Function to set mp3_root_dir, useful when script is used as module. '''
    global mp3_root_dir
    mp3_root_dir = new_root_dir

def set_npz_root_dir(new_root_dir):
    ''' Function to set npz_root_dir, useful when script is used as module. '''
    global npz_root_dir
    npz_root_dir = new_root_dir

def create_folder_structure():
    ''' Generate folder structure to store npz files. '''

    for dirpath, dirnames, filenames in os.walk(mp3_root_dir):
        structure = os.path.join(npz_root_dir, dirpath[len(mp3_root_dir):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("WARNING directory " + structure + " already exits! Are you sure it is empty?")

def mp3_path_to_npz_path(path):
    ''' Given the path of an mp3 file, returns the path of the npz file associated with it.
    
    Parameters
    ----------
    path : str
        The path of the mp3 file.

    Returns
    -------
    str
        The path of the npz file.
    
    Examples
    --------
    >>> set_mp3_root_dir('/User/Me/Desktop/7Digital')
    >>> set_npz_root_dir('/User/Me/Desktop/7Digital_NumPy')
    >>> path = '/User/Me/Desktop/7Digital/2/5/2573962.clip.mp3
    >>> mp3_path_to_npz_path(path)
    /User/Me/Desktop/7Digital_NumPy/2/5/2573962.npz
    '''

    return os.path.join(npz_root_dir, os.path.relpath(os.path.join(mp3_root_dir, path), mp3_root_dir))[:-9] + '.npz'

def savez(path):
    '''
    Parameters
    ----------
    path:
        The path of the mp3 file.
        
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

    path_npz = mp3_path_to_npz_path(path)
    array, sample_rate = librosa.core.load(path, sr=None, mono=False)
    array_split = librosa.effects.split(librosa.core.to_mono(array))
    np.savez(path_npz, array=array, sr=sample_rate, split=array_split)


def no_sound(df, start=0, end=None, verbose=True):
    '''
    Parameters
    ----------
        df: pd.DataFrame
            The input dataframe that stores the path of mp3s that will be converted to npz files.
            Recommendation: df = ultimate_output(df_fetch, discard_no_tag=True),
            where df_fetch is the dataframe from track_fetch.
    
        start: int
            The index of starting point in the pre_no_sound.csv.
            
        end: int
            The index of ending point in the pre_no_sound.csv.
            
        verbose: bool
            If True, print progress
         
    
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

    if end == None:
        end = len(df)
        
    start_time = time.time()
    tot = len(df)
    for idx, path in enumerate(df['path'][start:end]): 
        partial = time.time()
        path_npz = os.path.join(npz_root_dir, path[:-9] + '.npz')
        if os.path.isfile(path_npz):
            print("WARNING file " + path_npz + " already exists!") 
        else:
            path = os.path.join(mp3_root_dir, path)
            savez(path) 
        
        if verbose == True:
            if idx % 100 == 0:
                print('Processed {:6d} in {:8.4f} sec. Progress: {:2d}%'.format(idx, time.time() - start_time, int(idx / tot * 100)))
                print("Processed {} in {:6.5f} sec".format(path, time.time() - partial))

def no_sound_count(df, final_check=False):
    '''
    Parameters
    ----------
    df: pd.DataFrame
        The input dataframe that stores the path of mp3s that will be converted to npz files.
        Recommendation: df = track_wrangle.read_duplicates_and_purge() 
    
    final_check: bool
        final check mode.
        

        
    Returns
    -------
    l: list
        If it is in final check mode, it returns the list of path of the tracks
        whose mp3 has not been loaded and saved to numpy array.
        
    count: int
        Print progress.    
    '''
    
    count = 0
    l = []
     

    for idx, path in enumerate(df['path']):
        path_npz = os.path.join(npz_root_dir, path[:-9] + '.npz')
        if os.path.isfile(path_npz):
            count += 1
        elif final_check == True:
            l.append(path)
    
    if final_check == True:
        print("Processed {:6d} out of {:6d}...".format(count, len(df)))
        return l
    else:
        print("Processed {:6d} out of {:6d}...".format(count, len(df)))
        
if __name__ == "__main__":

    description= "Script to convert mp3 files into waveform NumPy arrays."
    epilog= "Example: python mp3_to_npz.py ./tracks_on_boden.csv --root-dir-npz /data/np_songs/"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("input", help="input csv filename or path")
    parser.add_argument("--root-dir-npz", help="set different directory to save npz files")
    parser.add_argument("--root-dir-mp3", help="set different directory to find mp3 files")

    args = parser.parse_args()
    
    if args.root_dir_npz:
        npz_root_dir = os.path.expanduser(args.root_dir_npz)
    if args.root_dir_mp3:
        mp3_root_dir = os.path.expanduser(args.root_dir_mp3)

    df = pd.read_csv(args.input)

    if os.path.isabs(df['path'][0]):
        mp3_root_dir_infer = os.path.dirname(os.path.commonprefix(df['path'].to_list()))
        if os.path.normpath(mp3_root_dir) != mp3_root_dir_infer:
            print('WARNING mp3_root_dir is different from what seems to be the right one given the input...')
            print('WARNING mp3_root_dir is now set as ' + mp3_root_dir_infer)
            mp3_root_dir = mp3_root_dir_infer
    
    create_folder_structure()

    no_sound(df)
    no_sound_count(df)