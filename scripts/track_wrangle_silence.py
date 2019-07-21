"""
"""
''' Contains tools for discarding mp3 files that are entirely or in major part silent


Notes
-----
This file can be run as a script. To do so, just type 'python mp3_to_npz.py' in the terminal. The help 
page should contain all the options you might possibly need. You will first need to run track_fetch.py (or
track_wrangle.py) in order have the right input file for this script, and mp3_to_npz.py in order to have
the right npz files ready.

This script performs the following operations:
    
- analysing results: check_silence analyse silent sections within the tracks and 
return a detail analysis dataframe.
- filtering results: return a DataFrame by filtering out tracks that are below a minimum 
trimmed length threshold, tracks that have total mid-silent section above a maximum 
threshold, and tracks that have length of maximum mid-silent section above a maximum 
threshold. See Glossary for what these terms mean.
    

Functions
---------
- check_silence
    Interpret the dataframe obtained from no_sound.py and add extra columns to return a new dataframe.
                              
- filter_trim_length
    Return the dataframe of tracks that satisfy the condition: tot length after trimming start/end silence >= threshold.
                              
- filter_tot_silence_duration
    Return the dataframe of tracks that satisfy the condition: tot length of mid-silence <= threshold.
                              
- filter_max_silence_duration
    Return the dataframe of tracks that satisfy the condition: max length of mid-silence section <= threshold.


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

import mp3_to_npz as npz

def check_silence(df, verbose=True): 
    '''
    Returns
    -------
    
    df: pd.DataFrame
        The dataframe is produced under the path provided in the Parameters. 
        This function extracts features from the npz files saved by the 
        mp3_to_npz.py. It combines several new columns based on the features 
        extracted and appended to the dataframe provided by the Parameters and 
        return a new csv with the extra columns.
        
        
        Extra columns:
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
    
    start = time.time()
    tot = len(df)

    audio_start = []
    audio_end = []
    mid_silence_length = []
    silence = []
    
    for idx, path in enumerate(df['file_path']):
        path_npz = npz.mp3_path_to_npz_path(path)
        try:
            ar = np.load(path_npz)
        except:
            print("WARNING at {:6d} out of {:6d}: {} was not savez'd correctly!".format(idx, len(df), path))
            npz.savez(path)
            ar = np.load(path_npz)
            
        sr = ar['sr']
        split = ar['split']
        split = split/sr
        
        audio_start.append(split[0,0])
        audio_end.append(split[-1,-1])

        bits = []
        bits_sum = 0

        for i in range(len(split) - 1):
            bit = (split[i,1], split[i+1,0])
            bits.append(bit)
            bits_sum += bit[1] - bit[0]
        
        silence.append(bits)
        mid_silence_length.append(bits_sum)
        
        if verbose == True:
            if idx % 100 == 0:
                print('Processed {:6d} in {:8.4f} s. Progress: {:2d}%'.format(idx, time.time() - start, int(idx / tot * 100)))
    
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
    cols = cols[:-9] + ['effective_clip_length', 'audio_start', 'audio_end'] + cols[-6:-4] + ['max_silence_length'] + cols[-4:-1]
    df = df[cols]
        
    if verbose == True:
        print('Processed {:6d} in {:8.4f} s.'.format(tot, time.time() - start))

    return df

def filter_trim_length(df, threshold):
    '''
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
    '''
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
    '''
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
    parser.add_argument("--filter-trim-length", type=float, default=0, help="keep only tracks whose effective length (in seconds) is longer than the theshold")
    parser.add_argument("--filter-tot-silence", type=float, default=0, help="keep only tracks whose tot silent length is shorter than the theshold")
    parser.add_argument("--filter-max-silence", type=float, default=0, help="keep only tracks whose max silent length is shorter than the theshold")

    args = parser.parse_args()

    if args.output[-4:] != '.csv':
        args.output = args.output + '.csv' 

    if os.path.isfile(args.output):
       print("WARNING file " + args.output + " already exists!")
       sys.exit(0)

    if args.root_dir_npz:
        npz.set_npz_root_dir(os.path.expanduser(args.root_dir_npz))
    if args.root_dir_mp3:
        npz.set_mp3_root_dir(os.path.expanduser(args.root_dir_mp3))

    df = pd.read_csv(args.input, comment='#')

    if os.path.isabs(df['file_path'][0]):
        mp3_root_dir_infer = os.path.dirname(os.path.commonprefix(df['file_path'].to_list()))
        if os.path.normpath(mp3_root_dir) != mp3_root_dir_infer:
            print('WARNING mp3_root_dir is different from what seems to be the right one given the input...')
            print('WARNING mp3_root_dir is now set as ' + mp3_root_dir_infer)
            mp3_root_dir = mp3_root_dir_infer

    df = check_silence(df)
    if args.filter_trim_length:
        df = filter_trim_length(df, args.filter_trim_length)
    if args.filter_tot_silence:
        df = filter_tot_silence_duration(df, args.filter_tot_silence)
    if args.filter_max_silence:
        df = filter_max_silence_duration(df, args.filter_max_silence)
    
    with open(args.output, 'a') as f:
        comment = '# python'
        comment += ' ' + os.path.basename(sys.argv.pop(0))
        for _ in range(len(sys.argv) - 2):
            comment += ' ' + sys.argv.pop(0)
        for _ in range(2):
            comment += ' ' + os.path.basename(sys.argv.pop(0))
        
        f.write(comment + '\n')

        df.to_csv(f, index=False)
