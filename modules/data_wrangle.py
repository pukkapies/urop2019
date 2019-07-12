'''
Davide Gallo (2019) Imperial College London
dg5018@ic.ac.uk

Copyright 2019, Davide Gallo <dg5018@ic.ac.uk>
'''

import h5py
import os
import pandas as pd

MP3_ROOT_DIR = '/srv/data/msd/7digital/'
OUTPUT_DIR = '/srv/data/urop/'
PATH_TO_H5 = '/srv/data/msd/msd_summary_file.h5'
PATH_TO_MISMATCHES_TXT = '/srv/data/msd/sid_mismatches.txt'
PATH_TO_DUPLICATES_TXT = '/srv/data/...'


### functions to fetch mp3 files on our server

def extract_ids_from_summary(file_path: str):
    with h5py.File(file_path, 'r') as h5:
        dataset_1 = h5['metadata']['songs']
        dataset_2 = h5['analysis']['songs']
        df_summary = pd.DataFrame(data={'track_7digitalid': dataset_1['track_7digitalid'], 'track_id': dataset_2['track_id']})
        df_summary['track_id'] = df_summary['track_id'].apply(lambda x: x.decode('UTF-8'))
        return df_summary

def find_tracks(root_dir: str):
    paths = []
    for folder, subfolders, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(os.path.abspath(folder), file)
            paths.append(path)
    # remove non-mp3 files from list
    paths = [path for path in paths if path[-4:] == '.mp3']
    return paths

def find_tracks_with_7dids(root_dir: str):
    paths = find_tracks(root_dir)
    paths_7dids = [int(os.path.basename(path)[:-9]) for path in paths]
    df = pd.DataFrame(data={'track_7digitalid': paths_7dids, 'path': paths})
    return df

def df_merge(track_summary_df: pd.DataFrame, track_df: pd.DataFrame):
    our_df = pd.merge(track_summary_df, track_df, on='track_7digitalid', how='inner')
    our_df = our_df[-our_df.duplicated('track_7digitalid', keep=False)]
    our_df = our_df[-our_df.duplicated('track_id', keep=False)]
    return our_df


### functions to purge mismatches

def df_purge_mismatches(track_df: pd.DataFrame, info_file: str):
    # generate a new dataframe with 'track_id' as index column, this makes searching through the index faster
    df = track_df.set_index('track_id')
    to_drop = []
    with open(info_file, 'r') as file:
        for line in file:
            to_drop.append(line[27:45])
    to_drop = [tid for tid in to_drop if tid in df.index]
    df.drop(to_drop, inplace=True)
    return df.reset_index()


### functions to purge mp3 errors in the files

def get_idx_mp3_size_zero(track_df: pd.DataFrame, root_dir: str):
    output = []
    for idx, path in enumerate(track_df['path']):
        path = os.path.join(root_dir, path)
        if os.path.getsize(path) == 0:
            output.append(idx)
        else:
            continue
    return output

def get_idx_mp3_size_less_than(track_df: pd.DataFrame, root_dir: str, threshold: int = 50000):
    output = []
    for idx, path in enumerate(track_df['path']):
        path = os.path.join(root_dir, path)
        if os.path.getsize(path) < threshold:
            output.append(idx)
        else:
            continue
    return output

def df_purge_faulty_mp3(track_df: pd.DataFrame, root_dir: str, threshold: int = 50000):
    if threshold == 0:
        return track_df.drop(get_idx_mp3_size_zero(track_df, root_dir))
    else:
        return track_df.drop(get_idx_mp3_size_less_than(track_df, root_dir, threshold))


### functions to purge tracks with no tags

def get_tids_with_tag() # to be moved on lastfm_query.py
    conn = sqlite3.connect('lastfm.db')
    q = "SELECT tid FROM tids"
    res = conn.execute(q)
    output = res.fetchall()
    output = [tid[0] for tid in output]
    conn.close()
    return output
    
def df_purge_without_tag(track_df: pd.DataFrame, db_path: str = None):
    # if db_path not specified, use default path (to be uncommented once get_tids_with_tags will be on lastfm_query.py)
    # if db_path:
    #    lastfm_query.set_path(db_path)

    tids_with_tag = get_tids_with_tag()
    tids_with_tag_df = pd.DataFrame(data={'track_id': tids_with_tag})
    
    return pd.merge(track_df, tids_with_tag_df, on='track_id', how='inner')


### functions to purge duplicates

from itertools import islice

def read_duplicates(info_file: str):
    l = []
    with open (info_file, 'r') as file:
        t = []
        for line in islice(file, 7, None):
            if line[0] == '%':
                l.append(t)
                t = []
            else:
                t.append(line[:18])
        l.append(t)
    return l

def read_duplicates_and_purge(track_df: pd.DataFrame, info_file: str):
    dups = read_duplicates(info_file)
    idxs = track_df.set_index('track_id').index
    dups_purged = [[tid for tid in sublist if tid in idx] for sublist in dups]
    return dups_purged

def df_purge_duplicates(track_df: pd.DataFrame, info_file:str, mode: str = 'single_random'):
    dups_purged = read_duplicates_and_purge(track_df, info_file)

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