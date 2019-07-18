''' Contains tools for fetching mp3 files on the server, matching 7digitalid's with tid's, and purging unwanted entries such as mismatches, faulty mp3 files, tracks without tags or duplicates


Notes
-----
This file can be run as a script. To do so, just type 'python track_wrangle.py' in the terminal. The help 
page should contain all the options you might possibly need. You will first need to run track_fetch.py and
provide the output of that script as an input argument for this one.

IMPORTANT: If using this script elsewhere than on Boden then rememer to use the option --root-dir to
set the directory in which the 7Digital mp3 files are stored.


Functions
---------
- set_path_h5                Sets path to the msd_summary_file.h5

- set_path_txt_mismatches    Sets path to the msd_mismatches.txt file

- set_path_txt_duplicates    Sets path to the msd_duplicates.txt file

- extract_ids_from_summary   Produces a dataframe with 7digitalid's and tid's of all tracks in the dataset

- find_tracks                Gets a list of paths of mp3 files on our server

- find_tracks_with_7dids     Produces a dataframe with 7digitalid's and paths of mp3 files on our server

- df_merge                   Produces a dataframe wiht 7digitalid's, tid's and paths of mp3 files on our server

- df_purge_mismatches        Removes mismatches from previously generated dataset

- df_purge_faulty_mp3_1      Removes tracks whose file size is less than the threshold (or have file size 0)

- df_purge_faulty_mp3_2      Removes tracks which can't be opened and therefore have NaN length

- df_purge_no_tag            Removes tracks which are not matched to any tag

- read_duplicates            Reads the msd_duplicates.txt file and produces a list (of lists) of duplicates

- df_purge_duplicates        Retains only one track for each set of duplicates

- ultimate_output            Combine all the previous functions and produces a dataframe accoring to the given parameters
'''

import h5py
import numpy as np
import os
import pandas as pd
import sys
# from mutagen import mp3 as mg
# from track_fetch import * # I am not importing a module, but rather taking the output of the script

from itertools import islice

if os.path.basename(os.getcwd()) == 'scripts':
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../modules')))
else:
    sys.path.insert(0, os.path.join(os.getcwd(), 'modules'))

import query_lastfm as db

path_h5 = '/srv/data/msd/msd_summary_file.h5'
path_txt_mismatches = '/srv/data/urop/msd_mismatches.txt'
path_txt_duplicates = '/srv/data/urop/msd_duplicates.txt'

def set_path_h5(new_path):
    global path_h5
    path_h5 = new_path

def set_path_txt_mismatches(new_path):
    global path_txt_mismatches
    path_txt_mismatches = new_path

def set_path_txt_duplicates(new_path):
    global path_txt_duplicates
    path_txt_duplicates = new_path

def extract_ids_from_summary(): # my mistake, this function should have always been here
    with h5py.File(path_h5, 'r') as h5:
        dataset_1 = h5['metadata']['songs']
        dataset_2 = h5['analysis']['songs']
        df_summary = pd.DataFrame(data={'track_7digitalid': dataset_1['track_7digitalid'], 'track_id': dataset_2['track_id']})
        df_summary['track_id'] = df_summary['track_id'].apply(lambda x: x.decode('UTF-8'))
        return df_summary

def df_merge(track_summary_df: pd.DataFrame, merged_df: pd.DataFrame):
    df = pd.merge(track_summary_df, merged_df, on='track_7digitalid', how='inner')
    df = df[-df.duplicated('track_7digitalid', keep=False)]
    df = df[-df.duplicated('track_id', keep=False)]
    return df

def df_purge_mismatches(merged_df: pd.DataFrame):
    df = merged_df.set_index('track_id')
    to_drop = []
    with open(path_txt_mismatches, 'r') as file: # I don't mind having regex here... I am happy with either
        for line in file:
            to_drop.append(line[27:45])
    to_drop = [tid for tid in to_drop if tid in df.index]
    df.drop(to_drop, inplace=True)
    return df.reset_index()

def df_purge_faulty_mp3_1(merged_df: pd.DataFrame, threshold: int = 0):
    df = merged_df[merged_df['file_size'] > threshold]
    return df

def df_purge_faulty_mp3_2(merged_df: pd.DataFrame):
    df = merged_df[-merged_df['track_length'].isna()]
    return df

def df_purge_no_tag(merged_df: pd.DataFrame, lastfm_db: str = None):
    if lastfm_db:
        db.set_path(lastfm_db)

    tids_with_tag = db.get_tids_with_tag()
    tids_with_tag_df = pd.DataFrame(data={'track_id': tids_with_tag})
    
    return pd.merge(merged_df, tids_with_tag_df, on='track_id', how='inner')

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

def df_purge_duplicates(merged_df: pd.DataFrame, randomness: bool = False):
    df = merged_df.set_index('track_id')
    dups = read_duplicates()
    idxs = df.index
    to_drop = [[tid for tid in sublist if tid in idxs] for sublist in dups]
    if not randomness:
        np.random.seed(42)
    for sublist in to_drop:
            if len(sublist) > 1:
                sublist.pop(np.random.randint(len(sublist)))
            else:
                continue
    to_drop = [tid for sublist in to_drop for tid in sublist]
    df.drop(to_drop, inplace=True)
    return df.reset_index()

def ultimate_output(df: pd.DataFrame, min_size: int = 0, min_length: int = 0, discard_no_tag: bool = False, discard_dupl: bool = False):
    ''' Produces a dataframe with the following columns: 'track_id', 'track_7digitalid', 'path', 'file_size', 'track_length', 'channels'.
    
    Parameters
    ----------
    min_size : int
        the tracks with file size (in bytes) below the threshold are discarded (50000 is a safe value)
    
    discard_no_tag : bool
        if True, discards tracks which are not matched to any tag

    discard_dupl : bool
        if True, discards tracks which are duplicates and keeps one for each set

    Returns
    -------
    df : pd.DataFrame
        The columns are the ones listed above.
        The entries are the ones specified by the given parameters.
    '''

    print("Fetching mp3 files from root directory...", end=" ")
    merged_df = df_merge(extract_ids_from_summary(), df)
    print("done")

    print("Purging mismatches...", end=" ")
    merged_df = df_purge_mismatches(merged_df)
    print("done")

    print("Purging faulty mp3 files...")
    print("    Checking files with size less than threshold...", end=" ")
    merged_df = df_purge_faulty_mp3_1(merged_df, threshold=min_size)
    print("done")
    print("    Checking files that can't be opened...", end=" ")
    merged_df = df_purge_faulty_mp3_2(merged_df)
    print("done")
    
    if discard_no_tag == True:
        print("Purging tracks with no tags...", end=" ")
        merged_df = df_purge_no_tag(merged_df)
        print("done")
    
    if discard_dupl == True:
        print("Purging duplicate tracks...", end=" ")
        df = df_purge_duplicates(merged_df)
        print("done")
    
    return merged_df

def die_with_usage():
    print()
    print("track_wrangle.py - Script to merge the list of mp3 files obtained with track_fetch.py with")
    print("                   the MSD summary file, remove unwanted entries such as mismatches, faulty")
    print("                   files or duplicates, and output a csv file with the following columns:")
    print("                   'track_id', 'track_7digitalid', 'path', 'track_length', 'file_size', 'channels'")
    print()
    print("Usage:     python track_wrangle.py <input csv filename or path> <output csv filename or path> [options]")
    print()
    print("General Options:")
    print("  --path-h5              Set path to msd_summary_file.h5.")
    print("  --path-txt-mism        Set path to mismatches info file.")
    print("  --path-txt-dupl        Set path to duplicates info file.")
    print("  --discard-no-tag       Choose to discard tracks with no tags.")
    print("  --discard-dupl         Choose to discard duplicate tracks.")
    print("  --help                 Show this help message and exit.")
    print("  --min-size             Set the minimum size (in bytes) to allow (default 0).")
    print()
    print("Example:   python track_wrangle.py /data/track_on_boden.csv ./wrangl.csv --min-size 50000 --discard-no-tag")
    print()
    sys.exit(0)

if __name__ == "__main__":

    if len(sys.argv) < 3:
        die_with_usage()
    
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
        
    min_size = 0
    discard_no_tag = False
    discard_dupl = False


    while True:
        if len(sys.argv) == 3:
            break
        elif sys.argv[3] == '--path-h5':
            set_path_h5(sys.argv[4])
            del sys.argv[3:5]
        elif sys.argv[3] == '--path-txt-mism':
            set_path_txt_mismatches(sys.argv[4])
            del sys.argv[3:5]
        elif sys.argv[3] == '--path-txt-dupl':
            set_path_txt_duplicates(sys.argv[4])
            del sys.argv[3:5]
        elif sys.argv[3] == '--min-size':
            min_size = int(sys.argv[3])
            del sys.argv[3:5]
        elif sys.argv[3] == '--discard-no-tag':
            discard_no_tag = True
            del sys.argv[3]
        elif sys.argv[3] == '--discard-dupl':
            discard_dupl = True
            del sys.argv[3]     
        else:
            print("???")
            sys.exit(0)

    df = pd.read_csv(sys.argv[1])
    df = ultimate_output(df, min_size, discard_no_tag, discard_dupl)
    df.to_csv(output, index=False)