# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:19:15 2019

@author: MacBook Pro
"""
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


import os
import pandas as pd
import numpy as np
import no_sound

if 'path_ult' not in globals():
    path_ult = '/srv/data/urop'

def set_path_ult_get_faulty_mp3(new_path):
    '''
    Parameters
    ----------
    
    
    new_path: str
        The path where 'ultimate_csv.csv' is stored.
        
    '''
    global path_ult
    path_ult = new_path
    
    
    

#path='D://UROP/millionsongsubset_full/MillionSongSubset'
def get_faulty_mp3(file='ultimate_csv_size.csv', track_id=True):
    
    '''
    Parameters
    ----------
    file: str
        The name of the input csv. Default--'ultimate_csv_size.csv'
        
    track_id: bool
        If True, the function will return track_ids of tracks, ortherwise
        track_7digitalid.
    
    
    
    Returns
    -------
    LIST: list
        A list of track_ids (or track_7digitalids) which could not be opened
        or have size zero.
        
    '''


    
    DIR = os.path.join(path_ult, file)
    df_size = pd.read_csv(DIR)
    
    #select tracks that were not opened correctly
    BOOL1 = df_size.loc[:,'lengths']>=999999999999
    
    #select tracks that have zero size
    BOOL2 = df_size.loc[:,'sizes']==0
    
    if track_id:
        LIST = df_size[BOOL1 | BOOL2].loc[:,"track_id"].tolist()
    else:
        LIST = df_size[BOOL1 | BOOL2].loc[:,"track_7digitalid"].tolist()
        
    return LIST



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
    
    df = no_sound.pre_no_sound()
    path_list = df.path.tolist()
    df_len = len(df)
    
    start_column = np.zeros(df_len)
    end_column = np.zeros(df_len)
    mid_silence_len = np.zeros(df_len)
    silence_detail = [None]*df_len
    
    for num, i in enumerate(path_list):
        path_np = '/srv/data/urop/7digital_numpy'+i[:-9]+'.npz'
        try:
            npz = np.load(path_np)
        except:
            print(i)
            no_sound.zip_correction(int(i[5:-9]))
            npz = np.load(path_np)
            
        sr = npz['sr']
        split = npz['split']
        
        #convert to seconds
        split = split/sr
        
        start_column[num] = split[0,0]
        end_column[num] = split[-1,-1]
        #calculate the total length of silence after starting and before ending
        mid_len = 0
        
        silence = []
        for j in range(len(split)-1):
            #silence section
            bit = (split[j,1], split[j+1,0])
            silence.append(bit)
            mid_len += bit[1]-bit[0]
        
        silence_detail[num] = silence
        mid_silence_len[num] = mid_len
        
        if num%100 ==0:
            print(num)
        
    df.loc[:,'start'] = start_column
    df.loc[:,'end'] = end_column
    df.loc[:,'silence_detail'] = silence_detail
    df.loc[:,'mid_silence_length'] = mid_silence_len
    df.loc[:, 'lengths(after_trim)'] = df.loc[:,'end']- df.loc[:,'start']
    df.loc[:,'non_silence_length'] = df.loc[:, 'lengths(after_trim)'] \
    - df.loc[:,'mid_silence_length']
    df.loc[:,'perc_after_trim'] = (df.loc[:,'non_silence_length'] \
          / df.loc[:, 'lengths(after_trim)'])*100
          
    def get_duration(x):
        LIST = []
        for i in x:
            if i != None:
                _ = i[1] - i[0]
                LIST.append(_)
        return LIST

    # get lengths of silent sections:
    df['silence_duration'] = df.silence_detail.apply(get_duration)
    
    def get_max(x):
        if x:
            return np.max(x)
        else:
            return None
    
    #get maximum lengths of individual silent sections:
    df['max_duration'] = df.silence_duration.apply(get_max)
    
    df.to_csv(os.path.join(path_ult, 'ultimate_csv_size2.csv'), index=False)
    
    


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


        
        
    

    