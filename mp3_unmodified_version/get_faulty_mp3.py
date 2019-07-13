# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:19:15 2019

@author: MacBook Pro
"""

'''
To use this function, you need to download the 'ultimate_csv_size.csv'. This
function takes the csv file and return the list of track_7digitalid of tracks
that either have zero size or cannot be opened.
If you want to return track_id instead, set track_7digitalid=False
'''


import os
import pandas as pd
import numpy as np
import no_sound

#path='D://UROP/millionsongsubset_full/MillionSongSubset'
def get_faulty_mp3(path, file='ultimate_csv_size.csv', track_id=True):
    '''
    set track_id=False if you want it to return track_7digitalid instead
    '''
    
    DIR = os.path.join(path, file)
    df_size = pd.read_csv(DIR)
    df_size = df_size.iloc[:,1:]
    
    #select tracks that were not opened correctly
    BOOL1 = df_size.loc[:,'lengths']>=999999999999
    
    #select tracks that have zero size
    BOOL2 = df_size.loc[:,'sizes']==0
    
    if track_id:
        LIST = df_size[BOOL1 | BOOL2].track_id.tolist()
    else:
        LIST = df_size[BOOL1 | BOOL2].track_id.tolist()
        
    return LIST



def check_silence(path='/home/aden/urop2019', file='pre_no_sound.csv'):
    
    DIR = os.path.join(path, file)
    df = pd.read_csv(DIR)
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
    
    df.to_csv(os.path.join(path, 'ultimate_csv_size2.csv'))
    
    
        
        
        
    

    