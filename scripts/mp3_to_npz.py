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

- progress check: no_sound_count() can be used to check progress of no_sound() on seperate window;


Examples
--------
    create_folder_structure()
    
    df = track_wrangle.read_duplicates_and_purge()
    
    no_sound(df, start=0, end=40000)
    
    no_sound_count()
    

Functions
---------
- set_mp3_root_dir              Tells the script the root directory of where mp3s were stored

- set_mpz_root_dir              Tells the script the root directory of where numpy arrays will be stored

- create_folder_structure       Copies the folder structure of how the mp3 files are saved and produce 
                                the same folder structure under a new directory, which will be used to save
                                the converted numpy arrays

- savez                         Converts a mp3 files into three numpy arrays as a single npz file:
                                    1. loaded mp3 arrays
                                    2. sampling rate 
                                    3. start and end position of arrays in 1. when volume of track is above 60dB (see librosa documentation on librosa.effect.split)

- no_sound                      Applies savez() to the provided provided by a dataframe

- no_sound_count                Returns the number of mp3 files that have been saved as npz files
                                Returns the path of tracks that have not been converted yet if final_check mode is True
                           
'''

import librosa
import numpy as np
import os
import pandas as pd
import sys
import time

mp3_root_dir = '/srv/data/msd/7digital/'
npz_root_dir = '/srv/data/urop/7digital_numpy/'

def set_mp3_root_dir(new_root_dir):
    global mp3_root_dir
    mp3_root_dir = new_root_dir

def set_npz_root_dir(new_root_dir):
    global npz_root_dir
    npz_root_dir = new_root_dir

def create_folder_structure():
    '''
    Generate folder structure to store npz files.
    '''
    for dirpath, dirnames, filenames in os.walk(mp3_root_dir):
        structure = os.path.join(npz_root_dir, dirpath[len(mp3_root_dir):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Directory " + structure + " already exits. Are you sure it is empty?")

def mp3_path_to_npz_path(path):
    return os.path.join(npz_root_dir, os.path.relpath(os.path.join(mp3_root_dir, path), mp3_root_dir))[:-9] + '.npz'

# def savez(track_7digitalid): # ADEN: the original code will make it more useful -- I actually used this for checking and fixing some individual errors
#     '''
#     Parameters
#     ----------
#     track_7digitalid: int
#         The track_7digitalid of the track 
        
#     Returns
#     -------
#     npz files:
#         The npz file is of the form ['array', 'sr', 'split']
#         'array': 
#             The loaded mp3 files in numpy-array form by library -- librosa.
#             The array represent the mp3s in its original sampling rate, and 
#             multi-channel is preserved. (each row represents one channel)
            
#         'sr':
#             The sampling rate of the mp3 file.
            
#         'split':
#             All the sections of the track which is non-silent (>=60dB). The
#             information is saved in the form: n*2 numpy.ndarray, and each row
#             represent one section -- starting position and ending position of 
#             array respectively.
    
#     '''
#     path = '/'+str(track_7digitalid)[0]+'/'+str(track_7digitalid)[1]+'/'+str(track_7digitalid)+'.clip.mp3' #
#     path_npz = npz_root_dir[:-1] +path[:-9] #
#     path = mp3_root_dir[:-1] +path #
#     array, sample_rate = librosa.core.load(path, sr=None, mono=False)
#     array_split = librosa.effects.split(librosa.core.to_mono(array))
#     np.savez(path_npz, array=array, sr=sample_rate, split=array_split)

# DAVIDE: I still think it is better to keep functions as generic as possible. This one is more clear in my opinion
def savez(path):
    path_npz = mp3_path_to_npz_path(path)
    array, sample_rate = librosa.core.load(path, sr=None, mono=False)
    array_split = librosa.effects.split(librosa.core.to_mono(array))
    np.savez(path_npz, array=array, sr=sample_rate, split=array_split)

def no_sound(df, start=0, end=501070, verbose=True):
    '''
    Parameters
    ----------
        df: pd.DataFrame
            The input dataframe that stores the path of mp3s that will be converted to npz files.
            Recommendation: df = track_wrangle.read_duplicates_and_purge()
    
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
    # paths = df['path'].tolist()[start:end]  #ADEN: this is probably more efficient
    # for idx, path in enumerate(paths)
    for idx, path in enumerate(df['path'][start:end]): # DAVIDE: the efficience gain is in milliseconds
        start = time.time()
        path_npz = os.path.join(npz_root_dir, path[:-9] + '.npz')   #ADEN: I think this is wrong # DAVIDE: it works
        if os.path.isfile(path_npz):
            print("File " + path_npz + " already exists. Ignoring.") 
        else:
            path = os.path.join(mp3_root_dir, path)
            # track_7digitalid = int(os.path.basename(path)[:-9])  #ADEN: since I changed savez
            # savez(track_7digitalid)
            savez(path) # DAVIDE: if we use my savez...
        
        if verbose == True:
            if idx % 100 == 0:
                print("{:6d} - time taken by {}: {}".format(idx, path, time.time() - start))
                

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
     
    # paths = df['path'].tolist()  # ADEN: This is probability more efficient. # DAVIDE the efficience gain is in milliseconds, and it is one line more of code
    # for idx, path in enumerate(paths):
    for idx, path in enumerate(df['path']):
        path_npz = npz_root_dir[:-1] + path[:-9]
        if os.path.isfile(path_npz + '.npz'):
            count += 1
        elif final_check == True:
            l.append(path)
    
    if final_check == True:
        print("    {} out of {} converted...".format(count, len(df)))
        return l
    else:
        print("    {} out of {} converted...".format(count, len(df)))
        
        

def die_with_usage():
    print()
    print("mp3_to_npz.py - Script to convert mp3 files into waveform NumPy arrays.")
    print()
    print("Usage:     python track_fetch.py <input filename> [options]")
    print()
    print("General Options:")
    print("  --root-dir-npz         Set different directory to save npz files.")
    print("  --root-dir-mp3         Set different directory to find mp3 files.")
    print("  --help                 Show this help message and exit.")
    print()
    print("Example:   python mp3_to_npz.py ./tracks_on_boden.csv --root-dir-npz /data/np_songs/")
    print()
    sys.exit(0)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        die_with_usage()
    
    if '--help' in sys.argv:
        if len(sys.argv) == 2:
            die_with_usage()
        else:
            print("???")
            sys.exit(0)

    while True:
        if len(sys.argv) == 2:
            break
        elif sys.argv[2] == '--root-dir-mp3':
            set_mp3_root_dir(os.path.expanduser(sys.argv[3]))
            del sys.argv[2:4]
        elif sys.argv[2] == '--root-dir-npz':
            set_npz_root_dir(os.path.expanduser(sys.argv[3]))
            del sys.argv[2:4]  
        else:
            print("???")
            sys.exit(0)

    df = pd.read_csv(sys.argv[1])

    mp3_root_dir_infer = os.path.dirname(os.path.commonprefix(df['path'].to_list()))
    
    if os.path.normpath(mp3_root_dir) != mp3_root_dir_infer:
        print('WARNING mp3_root_dir is different from what seems to be the right one given the input...')
        print('WARNING mp3_root_dir is now set as ' + mp3_root_dir_infer)
        set_mp3_root_dir(mp3_root_dir_infer)
    
    create_folder_structure()
    no_sound(df)
    no_sound_count(df)