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
----------

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

import numpy as np
import pandas as pd

import mp3_to_npz as npz


import numpy as np
import pandas as pd

import mp3_to_npz as npz

# mp3_root_dir = '/srv/data/msd/7digital/'
# npz_root_dir = '/srv/data/urop/7digital_numpy/'

# def set_mp3_root_dir(new_root_dir):
#     global mp3_root_dir
#     mp3_root_dir = new_root_dir

# def set_npz_root_dir(new_root_dir):
#     global npz_root_dir
#     npz_root_dir = new_root_dir

# def check_silence(df, save_csv=True, output_path='/srv/data/urop/ultimate_csv_size2.csv'): # ADEN: You will want to save a csv in here after the heavy computation and take a break...
def check_silence(df, verbose=True): # DAVIDE: same as in track_fetch.py...
                       #         check out 'if __name__ = __main__'; this script outputs a csv, there's no need to mention csv's in function declarations
    
    '''
    Parameters
    ----------
    path: str
        The path of the input dataframe.
        Recommendation: df = track_wrangle.read_duplicates_and_purge() 
    
        
    Returns
    -------
    
    df: pd.DataFrame
        The dataframe is produced under the path provided in the Parameters. 
        This function extracts features from the npz files saved by the 
        mp3_to_npz.py. It combines several new columns based on the features 
        extracted and appended to the dataframe provided by the Parameters and 
        return a new csv with the extra columns.
        
        
        Extra columns:
        'start': float
            The start time (s) of non-silent section of tracks.
            
        'end': float
            The end time (s) of non-silent section of tracks.
            
        'silent_details': list
            A list of tuples, and each tuple contains 2
            elements of the form (a,b). Each tuple represents a silent section 
            (after trimming), where a, b are the start and end time of the 
            section respectively.
            
        'mid_silence_length': float
            The sum of the length of all the mid-silent sections.
            
        'lengths(after_trim)': float
            The length of track after trimming.
            
        'non_silence_length': float
            The total duration of the non-silent sections of the track.
            
        'perc_after_trim': float
            The percentage of the track which is non-silent after trimming.
        
        'silence_duration': list
                length of each mid-silent section.
                
        'threshold': float
                maximum length of individual mid-silent sections from the 
                'silence_duration' column.
                
    csv: 
        The df is saved as csv if save_csv is True.
    
                
    '''
    
    tot = len(df)

    audio_start = []
    audio_end = []
    mid_silence_length = []
    silence = []
    
    for idx, path in enumerate(df['path']):
        #path_npz = os.path.join(npz_root_dir, path[:-9] + '.npz') # ADEN: was wrong, fixed it.
        #path_npz = npz_root_dir[:-1]+ path[:-9] + '.npz'
        #path_npz = os.path.join(npz_root_dir, path[:-9] + '.npz') # DAVIDE: it works! and os.path.join is ALWAYS safer than string concatenation
        path_npz = npz.mp3_path_to_npz_path(path)
        try:
            ar = np.load(path_npz)
        except:
            print("WARNING at {:6d} out of {:6d}: {} was not savez'd correctly!".format(idx, len(df), path))
            #track_7digitalid = int(os.path.basename(path)[:-9])  #ADEN: since I changed savez..
            #savez(path, path_npz) # DAVIDE: I still believe this sintax is more clear...
            #savez(track_7digitalid)
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
                print('Processed {:6d} out of {:6d}...'.format(idx, tot))
        
    df['audio_start'] = pd.Series(audio_start, index=df.index)
    df['audio_end'] = pd.Series(audio_end, index=df.index)
    df['effective_track_length'] = df['audio_end'] - df['audio_start']
    df['mid_silence_length'] = pd.Series(mid_silence_length, index=df.index)
    df['non_silence_length'] = df['effective_track_length'] - df['mid_silence_length']
    df['silence_percentage'] = df['mid_silence_length'] / df['effective_track_length'] * 100
    df['silence_detail'] = pd.Series(silence, index=df.index)
    df['silence_detail_length'] = df['silence_detail'].apply(lambda l: [i[1] - i[0] for i in l])
    df['max_silence_length'] = df['silence_detail_length'].apply(lambda l: [0] if l == [] else l).apply(lambda l: np.max(l)) # DAVIDE: check them out, both one-liner's ;)
        
    if verbose == True:
        print('Processed {:6d} out of {:6d}...'.format(tot, tot))

    # def get_duration(x): # DAVIDE: both added above...
    #     for i in x:
    #         if i != None:
    #             _ = i[1] - i[0]
    #             LIST.append(_)
    #     return LIST

    # df['silence_duration'] = df.silence.apply(get_duration)
    
    # def get_max(x):
    #     if x:
    #         return np.max(x)
    #     else:
    #         return None

    # df['threshold'] = df.silence_duration.apply(get_max)
            
    return df

def filter_trim_length(df, threshold):
    '''
    Parameters
    ---------
    df: pd.DataFrame
        The dataframe from the function check_silence.
    
    threshold: float
        If the length of tracks AFTER TRIMMING >= threshold, keep the track 
        and its track_id is returned.
    
    
    Returns
    ------
    list
        The track_id of tracks that satisfy the condition: 
            duration after trimming >= threshold.
    
    '''
    
    # ID = df[df['effective_track_length'] >= threshold]
    # return ID.track_id.tolist()
    return df[df['effective_track_length'] >= threshold] # DAVIDE: why not returning directly the dataframe?

def filter_tot_silence_duration(df, threshold):
    '''
    Parameters
    ---------
    df: pd.DataFrame
        The dataframe from the function check_silence.
    
    threshold: float
        If the sum of the length of mid-silent sections <= threshold, 
        keep the track and its track_id is returned.
    
    
    Returns
    ------
    list
        The track_id of tracks that satisfy the condition: 
            total length of mid-silent duration <= threshold.
    
    '''
    
    #ID = df[(df['mid_silence_length'] <= threshold)]
    #return ID.track_id.tolist()
    return df[df['mid_silence_length'] <= threshold] # DAVIDE: same as above

def filter_max_silence_duration(df, threshold):
    '''
    Parameters
    ---------
    df: pd.DataFrame
        The dataframe from the function check_silence.
    
    threshold: float
        If the maximum length amongst the individual mid-silent sections 
        <= threshold, keep the track and its track_id is returned.
    
    
    Returns
    ------
    list
        The track_id of tracks that satisfy the condition: 
            the maximum length amongst the individual mid-silent sections 
            <= threshold.
    
    '''
    
    # ID = df[(df['max_silence_length'] <= threshold) 
    #         | (np.isnan(df['threshold']))]
    # return ID.track_id.tolist()
    return df[df['max_silence_length'] <= threshold] # DAVIDE: same as above

if __name__ == "__main__":

    description = "Script to analyze npz arrays to extract information about silence."
    epilog = "Example: python track_wrangle_silence.py ./tracks_on_boden.csv --root-dir-npz /data/np_songs/"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("input", help="Input csv filename or path")
    parser.add_argument("output", help="Output csv filename or path")
    parser.add_argument("--root-dir-npz", help="Set different directory to save npz files.")
    parser.add_argument("--root-dir-mp3", help="Set different directory to find mp3 files.")
    parser.add_argument("--filter-trim-length", type=int, default=0, help="Keep only tracks whose effective length (in seconds) is longer than the theshold.")
    parser.add_argument("--filter-tot-silence-duration", type=int, default=0, help="Keep only tracks whose total silent length is shorter than the theshold.")
    parser.add_argument("--filter-max-silence-duration", type=int, default=0, help="Keep only tracks whose maximal silent length is shorter than the theshold.")

    args = parser.parse_args()

    if args.output[-4:] != '.csv':
        args.output = args.output + '.csv' 

    if args.root_dir_npz:
        npz.set_npz_root_dir(os.path.expanduser(args.root_dir_npz))
    if args.root_dir_mp3:
        npz.set_mp3_root_dir(os.path.expanduser(args.root_dir_mp3))

    df = pd.read_csv(args.input)

    mp3_root_dir_infer = os.path.dirname(os.path.commonprefix(df['path'].to_list()))
    
    if os.path.normpath(npz.mp3_root_dir) != mp3_root_dir_infer:
        print('WARNING mp3_root_dir is different from what seems to be the right one given the input...')
        print('WARNING mp3_root_dir is now set as ' + mp3_root_dir_infer)
        npz.set_mp3_root_dir(mp3_root_dir_infer)

    df = check_silence(df)
    if args.filter_trim_length:
        df = filter_trim_length(df, args.filter_trim_length)
    if args.filter_tot_silence:
        df = filter_tot_silence_duration(df, args.filter_tot_silence)
    if args.filter_max_silence:
        df = filter_max_silence_duration(df, args.filter_max_silence)
    df.to_csv(args.output, index=False)
