'''
Note
----
This module interprets the result obtained from no_sound.py. The structure can
be divided into two steps:
    
-Analyse result: get_faulty_mp3() checks for broken mp3 tracks, and 
check_silence analyse silent sections within the tracks and return a detail 
analysis dataframe.
-Filter result: Return a DataFrame by filtering out tracks that are below 
a minimum trimmed length threshold, tracks that have total mid-silent section
above a maximum threshold, and tracks that have length of maximum mid=silent
section above a maximum threshold. See Glossary for what the terms mean.
    


Functions
---------

- set_path_ult
    Tell the script the path of where 'ultimate_csv.csv' was stored.
    
- get_faulty_mp3
    Return a list of mp3 which cannot be opened or have size zero.

- check_silence
    Interpret the csv obtained from no_sound.py and add extra columns to the 
    output csv.
                              
- filter_trim_length          
    Return the track_id of tracks that satisfy the condition: 
    duration after trimming >= min_duration.
                              
- filter_tot_silence_duration 
    Return the track_id of tracks that satisfy the condition: 
    total length of mid-silent duration <= max_duration.
                              
- filter_max_silence_duration 
    Return the track_id of tracks that satisfy the condition: 
    the maximum length amongst the individual mid-silent 
    sections <= max_duration.



Glossary
----------

    trim:
        The total duration of tracks excluding the starting and ending section 
        if they are silent respectively.
        
    mid-silent section:
        Any section in the track which is silent, but it is neither the 
        starting silent section nor the ending silent section.
        

Examples
--------
    import pd
    
    df_pre = no_sound.pre_no_sound()
    
    df = check_silence(df_pre)   /or/   
    df = pd.read_csv(os.path.join(path_ult, 'ultimate_csv_size2.csv'))

    a = filter_tot_silence_duration(df, 1)
    
    b = filter_max_silence_duration(df, 1)
    
    c = filter_trim_length(df, 15)

    set(a).intersection(set(b), set(c))  #return list of track_ids
'''

import numpy as np
import os
import pandas as pd
import sys

from mp3_to_npz import savez

mp3_root_dir = '/srv/data/msd/7digital/'
npz_root_dir = '/srv/data/urop/7digital_numpy/'

def set_mp3_root_dir(new_root_dir):
    global mp3_root_dir
    mp3_root_dir = new_root_dir

def set_npz_root_dir(new_root_dir):
    global npz_root_dir
    npz_root_dir = new_root_dir

def check_silence(df):
    
    '''
    Parameters
    ----------
    path: str
        The path where the csv returned by the function pre_no_sound is saved.
    
    file: str
        The saved name of the csv returned by the function pre_no_sound.
        
    Returns
    -------
    csv:
        The csv is saved under the path provided in the Parameters. This
        function extracts features from the npz files saved by the no_sound.py.
        It combines several new columns based on the features extracted and
        appended to the dataframe provided by the Parameters and return a new 
        csv with the extra columns.
        
        
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
                
        'max_duration': float
                maximum length of individual mid-silent sections from the 
                'silence_duration' column.
    
                
    '''
    
    audio_start = []
    audio_end = []
    mid_silence_length = []
    silence = []
    
    for idx, path in enumerate(df['path']):
        path_npz = os.path.join(npz_root_dir, path[:-9] + '.npz')
        try:
            npz = np.load(path_npz)
        except:
            print(idx)
            savez(path, path_npz)
            npz = np.load(path_npz)
            
        sr = npz['sr']
        split = npz['split']
        
        #convert to seconds
        split = split/sr
        
        audio_start.append(split[0,0])
        audio_end.append(split[-1,-1])

        #calculate the total length of silence after starting and before ending
        bits = []
        bits_sum = 0

        for i in range(len(split) - 1):
            # silence section
            bit = (split[i,1], split[i+1,0])
            bits.append(bit)
            bits_sum += bit[1] - bit[0]
        
        silence.append(bits)
        mid_silence_length.append(bits_sum)
        
        if idx % 100 == 0:
            print(idx)
        
    df['audio_start'] = pd.Series(audio_start, index=df.index)
    df['audio_end'] = pd.Series(audio_end, index=df.index)
    df['mid_silence_length'] = pd.Series(mid_silence_length, index=df.index)
    df['effective_length'] = df['audio_end'] - df['audio_start']
    df['non_silence_length'] = df['effective_length'] - df['mid_silence_length']
    df['silence_percentage'] = df['non_silence_length'] / df['effective_length'] * 100
    df['silence_detail'] = pd.Series(silence, index=df.index)

    return df
          
    # def get_duration(x):
    #     LIST = []
    #     for i in x:
    #         if i != None:
    #             _ = i[1] - i[0]
    #             LIST.append(_)
    #     return LIST

    # # get lengths of silent sections:
    # df['silence_duration'] = df.silence.apply(get_duration)
    
    # def get_max(x):
    #     if x:
    #         return np.max(x)
    #     else:
    #         return None
    
    # #get maximum lengths of individual silent sections:
    # df['max_duration'] = df.silence_duration.apply(get_max)
    
    


def filter_trim_length(df, min_duration):
    '''
    Parameters
    ---------
    df: pd.DataFrame
        The datafram from the function filter_mp3.
    
    min_duration: float
        If the length of tracks AFTER TRIMMING >= min_duration, keep the track 
        and its track_id is returned.
    
    
    Returns
    ------
    list
        The track_id of tracks that satisfy the condition: 
            duration after trimming >= min_duration.
    
    '''
    
    ID = df[df['lengths(after_trim)'] >= min_duration]
    return ID.track_id.tolist()



def filter_tot_silence_duration(df, max_duration):
    '''
    Parameters
    ---------
    df: pd.DataFrame
        The datafram from the function filter_mp3.
    
    max_duration: float
        If the sum of the length of mid-silent sections <= min_duration, 
        keep the track and its track_id is returned.
    
    
    Returns
    ------
    list
        The track_id of tracks that satisfy the condition: 
            total length of mid-silent duration <= max_duration.
    
    '''
    
    ID = df[(df['mid_silence_length'] <= max_duration)]
    return ID.track_id.tolist()


#This functions takes the dataframe and max_duration of an individual mid-silent section (e.g. 1s)
#return the track_id that SATISFIES the condition -- the maximum length of an indiviual mid-silent section <= max_duration
def filter_max_silence_duration(df, max_duration):
    '''
    Parameters
    ---------
    df: pd.DataFrame
        The datafram from the function filter_mp3.
    
    max_duration: float
        If the maximum length amongst the individual mid-silent sections 
        <= max_duration, keep the track and its track_id is returned.
    
    
    Returns
    ------
    list
        The track_id of tracks that satisfy the condition: 
            the maximum length amongst the individual mid-silent sections 
            <= max_duration.
    
    '''
    
    ID = df[(df['max_duration'] <= max_duration) 
            | (np.isnan(df['max_duration']))]
    return ID.track_id.tolist()

def die_with_usage():
    print()
    print("NOOOOO: mp3_to_npz.py - Script to convert MP3 files into waveform NumPy arrays.")
    print()
    print("Usage:     python track_fetch.py <input filename> [options]")
    print()
    print("General Options:")
    print("  --root-dir-npz         Set different directory to save npz files.")
    print("  --root-dir-mp3         Set different directory to find mp3 files.")
    print("  --help                 Show this help message and exit.")
    print("  --verbose              Show progress.")
    print()
    print("Example:   python mp3_to_npz.py ./tracks_on_boden.csv --root-dir-npz /data/np_songs/")
    print()
    sys.exit(0)

if __name__ == "__main__":

    # show help
    if len(sys.argv) < 3:
        die_with_usage()
    
    # show help, if user did not input something weird
    if '--help' in sys.argv:
        if len(sys.argv) == 2:
            die_with_usage()
        else:
            print("???")
            sys.exit(0)

    if sys.argv[2][-4:] == '.csv':
        output = sys.argv[2]
    else:
        output = sys.argv[2] + '.csv'
    
    verbose = False

    while True:
        if len(sys.argv) == 3:
            break
        elif sys.argv[3] == '--root-dir-mp3':
            set_mp3_root_dir(sys.argv[4])
            del sys.argv[3:5]
        elif sys.argv[3] == '--root-dir-npz':
            set_npz_root_dir(sys.argv[4])
            del sys.argv[3:5]
        elif sys.argv[3] == '--verbose':
            verbose = True
            del sys.argv[3]     
        else:
            print("???")
            sys.exit(0)

    df = pd.read_csv(sys.argv[1])
    df = check_silence(df)
    df.to_csv(output, index=False)