''' Contains tools for discarding mp3 files that are entirely or in major part silent


Notes
-----
This file can be run as a script. To show parameters, just type 'python mp3_to_npz.py --help' in the terminal. The help 
page should contain all the options you might possibly need. You will first need to run track_fetch.py (or
track_wrangle.py) in order have the right input file for this script, and mp3_to_npz.py in order to have
the right npz files ready.

This script performs the following operations:
    
- analysing results: 
    check_silence analyse silent sections within the tracks and return a detail 
    analysis dataframe.

- filtering results: 
    return a DataFrame by filtering out tracks that are 
    below a minimum trimmed length threshold, tracks that have total mid-silent 
    section above a maximum threshold, and tracks that have length of maximum 
    mid-silent section above a maximum threshold. See Glossary for what these 
    terms mean.
    

Functions
---------
- check_silence
    Interpret the npz files generated by mp3_to_npz.py and add extra information to the input dataframe.
                              
- filter_trim_length
    Remove entries from dataframe with an effective clip length smaller than a specified threshold.
                              
- filter_tot_silence_duration
    Removes entries from dataframe with total silence longer than a specified threshold.
                              
- filter_max_silence_duration
    Removes entries from dataframe with longest silent segment longer than a specified threshold.


Glossary
--------
    trim:
        The total duration of tracks excluding the starting and ending section 
        if they are silent.
        
    mid-silent section:
        Any section in the track which is silent, but it is neither the starting silent section
        nor the ending silent section.
'''

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.realpath(__file__))))

import mp3_to_numpy as npz

def check_silence(df, verbose=True): 
    ''' Extracts silence-related features from .npz files and adds these to given DataFrame.


    Parameters
    ----------
    df: pd.DataFrame

    verbose: bool
        Specifies wether to print progress or not.

    Returns
    -------
    df : pd.DataFrame
        DataFrame given as parameter with the following extra columns:

        'effective_clip_length': float
            The length of the track after trimming.
        
        'audio_start': float
            The start time (in s) of non-silent section in the tracks.
            
        'audio_end': float
            The end time (in s) of non-silent section in the tracks.
            
        'mid_silence_length': float
            The sum of the length of all the mid-silent sections.
            
        'non_silence_length': float
            The total duration of the non-silent sections of the track.

        'max_silence_length': float
            The maximal length of individual mid-silent sections from the 'silence_duration' column.
        
        'silence_detail_length': list
            The length of each mid-silent section.

        'silence_detail': list
            A list of tuples, each tuple containing 2 elements of the form (a,b), each tuple
            representing a silent section (after trimming). 
            Here a and b are the start and end time of the section (respectively).

        'silence_percentage': float
            The percentage of the track which is non-silent after trimming.
    '''
   
    if verbose:
        start = time.time()
        tot = len(df)

    npz_paths = []

    # initialize silence details
    audio_start = []
    audio_end = []
    mid_silence_length = []
    silence = []

    for idx, path in enumerate(df['file_path']):
        npz_path = npz.mp3_path_to_npz_path(path)
        npz_paths.append(os.path.relpath(npz_path, npz.npz_root_dir))
        
        # try to load the stored npz file
        try:
            ar = np.load(npz_path)
        # if file cannot be loaded, re-save the mp3 as npz and load the npz again
        except:
            print("WARNING at {:6d} out of {:6d}: {} was not savez'd correctly!".format(idx, len(df), path))
            npz.savez(path, npz_path)
            ar = np.load(npz_path)
        
        # load info stored in the npz file
        sr = ar['sr']
        split = ar['split']
        
        # convert sampling rate into second
        split = split/sr
        
        # retrieve the starting time of the non-silent section of track
        audio_start.append(split[0,0])
        
        # retrieve the ending time of the non-silent section of track
        audio_end.append(split[-1,-1])

        # store info about time interval of the mid-silent sections
        bits = []
        bits_sum = 0

        for i in range(len(split) - 1): 
            bit = (split[i,1], split[i+1,0]) # tuple of the form (start of silent section, end of silent section)
            bits.append(bit)
            
            # record the sum of length of mid-silent sections
            bits_sum += bit[1] - bit[0]
        
        # store info about silence interval 
        silence.append(bits)
        
        # store sum of length of mid-silent section
        mid_silence_length.append(bits_sum)
        
        # print progress
        if verbose:
            if idx % 100 == 0:
                print('Processed {:6d} in {:8.4f} s. Progress: {:2d}%'.format(idx, time.time() - start, int(idx / tot * 100)))
    
    # append new columns
    df['mp3_path'] = df['file_path']
    df['npz_path'] = pd.Series(npz_paths, index=df.index)
    df['audio_start'] = pd.Series(audio_start, index=df.index)
    df['audio_end'] = pd.Series(audio_end, index=df.index)
    df['effective_clip_length'] = df['audio_end'] - df['audio_start']
    df['mid_silence_length'] = pd.Series(mid_silence_length, index=df.index)
    df['non_silence_length'] = df['effective_clip_length'] - df['mid_silence_length']
    df['silence_detail'] = pd.Series(silence, index=df.index)
    df['silence_detail_length'] = df['silence_detail'].apply(lambda l: [i[1] - i[0] for i in l])
    df['silence_percentage'] = df['mid_silence_length'] / df['effective_clip_length'] * 100
    df['max_silence_length'] = df['silence_detail_length'].apply(lambda l: [0] if l == [] else l).apply(lambda l: np.max(l))

    cols = df.columns.tolist()
    
    # rearrange order of columns
    cols = cols[:-11] + ['effective_clip_length', 'audio_start', 'audio_end', 'mid_silence_length', 'non_silence_length', 'max_silence_length', 'silence_detail', 'silence_detail_length', 'silence_percentage']
    cols.remove('file_path')
    cols.insert(3,'npz_path')
    cols.insert(3,'mp3_path')
    df = df[cols]
    
    if verbose:
        print('Processed {:6d} in {:8.4f} s.'.format(tot, time.time() - start))

    return df

def filter_trim_length(df, threshold):
    ''' Remove entries from dataframe with an effective clip length smaller than a specified threshold.

    Parameters
    ---------
    df: pd.DataFrame
        The dataframe from the function check_silence.
    
    threshold: float
        If the length of tracks AFTER TRIMMING >= threshold, keep the track.
    
    
    Returns
    ------
    df: pd.DataFrame
        The rows that satisfy the condition: 
            duration after trimming >= threshold.
    
    '''
    
    return df[df['effective_clip_length'] >= threshold] 

def filter_tot_silence_duration(df, threshold):
    ''' Removes entries from dataframe with total silence longer than a specified threshold.

    Parameters
    ---------
    df: pd.DataFrame
        The dataframe from the function check_silence.
    
    threshold: float
        If the sum of the length of mid-silent sections <= threshold, keep the track and its track_id is returned.
    
    Returns
    ------
    df: pd.DataFrame
        The rows that satisfy the condition: 
            tot length of mid-silent duration <= threshold.
    
    '''
    
    return df[df['mid_silence_length'] <= threshold] 

def filter_max_silence_duration(df, threshold):
    ''' Removes entries from dataframe with longest silent segment longer than a specified threshold.

    Parameters
    ---------
    df: pd.DataFrame
        The dataframe from the function check_silence.
    
    threshold: float
        If the maximum length amongst the individual mid-silent sections <= threshold, keep the track.
    
    Returns
    ------
    df: pd.DataFrame
        The rows that satisfy the condition: 
            max length amongst the individual mid-silent sections <= threshold.
    
    '''
    
    return df[df['max_silence_length'] <= threshold]

if __name__ == "__main__":

    description = "Script to analyze npz arrays to extract information about silence."
    epilog = "Example: python track_wrangle_silence.py ./tracks_on_boden.csv --root-dir-npz /data/np_songs/"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("input", help="input csv filename or path")
    parser.add_argument("output", help="output csv filename or path")
    parser.add_argument("--root-dir-npz", help="set directory to save npz files")
    parser.add_argument("--root-dir-mp3", help="set directory to find mp3 files")
    parser.add_argument("--min-size", type=int, default=0, help="set the minimum size (in bytes) to allow")
    parser.add_argument("--filter-trim-length", type=float, default=0, help="keep only tracks whose effective length (in seconds) is longer than the theshold")
    parser.add_argument("--filter-tot-silence", type=float, default=0, help="keep only tracks whose tot silent length is shorter than the theshold")
    parser.add_argument("--filter-max-silence", type=float, default=0, help="keep only tracks whose max silent length is shorter than the theshold")

    args = parser.parse_args()

    if args.output[-4:] != '.csv':
        output = args.output + '.csv'
    else:
        output = args.output

    if os.path.isfile(output):
       print("WARNING file " + output + " already exists!")
       sys.exit(0)

    df = pd.read_csv(args.input, comment='#')

    assert 'file_size' in df
    
    need_check_silence = False

    cols = ['effective_clip_length', 'audio_start', 'audio_end', 'mid_silence_length', 'non_silence_length', 'max_silence_length', 'silence_detail_length', 'silence_detail', 'silence_percentage']

    # if any column already exists in the df, dont run check_silence (i.e. adding the columns) 
    if [col for col in cols if col in df.columns] != cols:
        if args.root_dir_npz:
            npz.set_npz_root_dir(os.path.abspath(os.path.expanduser(args.root_dir_npz)))
        if args.root_dir_mp3:
            npz.set_mp3_root_dir(os.path.abspath(os.path.expanduser(args.root_dir_mp3)))
        need_check_silence = True
    
    elif not any([args.filter_trim_length, args.filter_tot_silence, args.filter_max_silence, args.min_size]):
        print("Nothing to be done!!")
        sys.exit(0)

    if need_check_silence:
        df = check_silence(df)        
    if args.filter_trim_length:
        df = filter_trim_length(df, args.filter_trim_length)
    if args.filter_tot_silence:
        df = filter_tot_silence_duration(df, args.filter_tot_silence)
    if args.filter_max_silence:
        df = filter_max_silence_duration(df, args.filter_max_silence)

    df = df[df['file_size'] > args.min_size]
    
    # create output csv file
    with open(output, 'a') as f:
        # insert comment line displaying options used
        comment = '# python'
        comment += ' ' + os.path.basename(sys.argv.pop(0))
        
        if need_check_silence:
            comment += ' --root-dir-npz ' + npz.npz_root_dir + ' --root-dir-mp3 ' + npz.mp3_root_dir
        
        options = [arg for arg in sys.argv if arg not in (args.input, args.output, '--root-dir-npz', npz.npz_root_dir, '--root-dir-mp3', npz.mp3_root_dir)]
        for option in options:
            comment += ' ' + option
        
        comment += ' ' + os.path.basename(args.input) + ' ' + os.path.basename(output)
        
        # write comment to the top line
        f.write(comment + '\n')
        
        # write dataframe
        df.to_csv(f, index=False)
