# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:23:42 2019

@author: MacBook Pro
"""

import os
import pandas as pd
import sys
#This is the path where your hdf5_getter.py file sits
sys.path.insert(0, 'C://Users/MacBook Pro/UROP2019/')


import get_faulty_mp3

path='D://UROP/millionsongsubset_full/MillionSongSubset'
file = 'ultimate_csv_size.csv'

df_size = pd.read_csv(os.path.join(path, file))

df_test = pd.read_csv(os.path.join('D://UROP', 'lastfm_tags_tids.csv'))

df_final = df_size.merge(df_test, left_on='track_7digitalid', right_on='tid')

fault = get_faulty_mp3.get_faulty_mp3(path)

df_final = df_final[-df_final.track_7digitalid.isin(fault)]

df_final = df_final.iloc[:,1:]
df_final = df_final.drop(['tid'], axis=1)

df_final = df_final.sort_values('path')

df_final.to_csv(os.path.join(path, 'pre_no_sound.csv'), index_label=False)

