'''
Davide Gallo (2019) Imperial College London
dg5018@ic.ac.uk

Copyright 2019, Davide Gallo <dg5018@ic.ac.uk>
'''

import h5py
import os
import pandas as pd

MP3_ROOT_DIR = '/srv/data/msd/7digital/'

path_h5 = '/srv/data/msd/msd_summary_file.h5'
path_txt_mismatches = '/srv/data/msd/sid_mismatches.txt'
path_txt_duplicates = '/srv/data/...'

def set_path_h5(new_path):
    global path_h5
    path_h5 = new_path

def set_path_txt_mismatches(new_path):
    global path_txt_mismatches
    path_txt_mismatches = new_path

def set_path_txt_duplicates(new_path):
    global path_txt_duplicates
    path_txt_duplicates = new_path

### functions to fetch mp3 files on our server

def extract_ids_from_summary():
    with h5py.File(path_h5, 'r') as h5:
        dataset_1 = h5['metadata']['songs']
        dataset_2 = h5['analysis']['songs']
        df_summary = pd.DataFrame(data={'track_7digitalid': dataset_1['track_7digitalid'], 'track_id': dataset_2['track_id']})
        df_summary['track_id'] = df_summary['track_id'].apply(lambda x: x.decode('UTF-8'))
        return df_summary

def find_tracks():
    paths = []
    for folder, subfolders, files in os.walk(MP3_ROOT_DIR):
        for file in files:
            path = os.path.join(os.path.abspath(folder), file)
            paths.append(path)
    # remove non-mp3 files from list
    paths = [path for path in paths if path[-4:] == '.mp3']
    return paths

def find_tracks_with_7dids():
    paths = find_tracks()
    paths_7dids = [int(os.path.basename(path)[:-9]) for path in paths]
    df = pd.DataFrame(data={'track_7digitalid': paths_7dids, 'path': paths})
    return df

def df_merge(track_summary_df: pd.DataFrame, track_df: pd.DataFrame):
    our_df = pd.merge(track_summary_df, track_df, on='track_7digitalid', how='inner')
    our_df = our_df[-our_df.duplicated('track_7digitalid', keep=False)]
    our_df = our_df[-our_df.duplicated('track_id', keep=False)]
    return our_df


### functions to purge mismatches

def df_purge_mismatches(track_df: pd.DataFrame):
    # generate a new dataframe with 'track_id' as index column, this makes searching through the index faster
    df = track_df.set_index('track_id')
    to_drop = []
    with open(path_txt_mismatches, 'r') as file:
        for line in file:
            to_drop.append(line[27:45])
    to_drop = [tid for tid in to_drop if tid in df.index]
    df.drop(to_drop, inplace=True)
    return df.reset_index()


### functions to purge mp3 errors in the files

def get_idx_mp3_size_zero(track_df: pd.DataFrame):
    output = []
    for idx, path in enumerate(track_df['path']):
        path = os.path.join(MP3_ROOT_DIR, path)
        if os.path.getsize(path) == 0:
            output.append(idx)
        else:
            continue
    return output

def get_idx_mp3_size_less_than(track_df: pd.DataFrame, threshold: int = 50000):
    output = []
    for idx, path in enumerate(track_df['path']):
        path = os.path.join(MP3_ROOT_DIR, path)
        if os.path.getsize(path) < threshold:
            output.append(idx)
        else:
            continue
    return output

def df_purge_faulty_mp3(track_df: pd.DataFrame, threshold: int = 50000):
    if threshold == 0:
        return track_df.drop(get_idx_mp3_size_zero(track_df))
    else:
        return track_df.drop(get_idx_mp3_size_less_than(track_df, threshold))


### functions to purge tracks with no tags

import lastfm_query
    
def df_purge_without_tag(track_df: pd.DataFrame, lastfm_db: str = None):
    # if db_path not specified, use default path (to be uncommented once get_tids_with_tags will be on lastfm_query.py)
    if lastfm_db:
        lastfm_query.set_path(lastfm_db)

    tids_with_tag = get_tids_with_tag()
    tids_with_tag_df = pd.DataFrame(data={'track_id': tids_with_tag})
    
    return pd.merge(track_df, tids_with_tag_df, on='track_id', how='inner')


### functions to purge duplicates

from itertools import islice

def read_duplicates():
    l = []
    with open (path_txt_duplicates, 'r') as file:
        t = []
        for line in islice(file, 7, None):
            if line[0] == '%':
                l.append(t)
                t = []
            else:
                t.append(line[:18])
        l.append(t)
    return l

def df_purge_duplicates(track_df: pd.DataFrame, mode: str = 'single_random'):
    dups = read_duplicates()
    idxs = track_df.set_index('track_id').index
    dups_purged = [[tid for tid in sublist if tid in idxs] for sublist in dups]

    if mode == 'single_random': # currently not at all random: keeps only last track
        df = track_df.set_index('track_id')
        to_drop = dups_purged
        for subset in to_drop:
            subset.pop()
        to_drop = [tid for sublist in to_drop for tid in sublist]
        df.drop(to_drop, inplace=True)
        return df.reset_index()
    
    else:
        raise KeyError

### output

def ultimate_output(threshold: int = 50000, discard_no_tag: bool = True, discard_dupl: bool = True, discard_dupl_mode: str = 'single_random'):
    df = df_merge(extract_ids_from_summary(), find_tracks_with_7dids())
    df = df_purge_mismatches(df)
    df = df_purge_faulty_mp3(df, threshold=threshold)
    
    if discard_no_tag == True:
        df = df_purge_without_tag(df)
    
    if discard_dupl == True:
        df = df_purge_duplicates(df, discard_dupl_mode)
    
    return df

### output

def read_duplicates_and_purge(threshold: int = 50000, discard_no_tag: bool = True): # standalone function; not used to generate the ultimate_output()
    df = df_merge(extract_ids_from_summary(), find_tracks_with_7dids())
    df = df_purge_mismatches(df)
    df = df_purge_faulty_mp3(df, threshold=threshold)
    
    if discard_no_tag == True:
        df = df_purge_without_tag(df)

    dups = read_duplicates()
    idxs = df.set_index('track_id').index
    dups_purged = [[tid for tid in sublist if tid in idxs] for sublist in dups]
    return dups_purged