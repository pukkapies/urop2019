''' Contains tools for fetching mp3 files on the server, matching 7digitalid's with tid's, and purging unwanted entries such as mismatches, faulty mp3 files, tracks without tags or duplicates.


Notes
-----
This file can be run as a script. To do so, just type 'python track_wrangle.py' in the terminal. The help 
page should contain all the options you might possibly need. You will first need to run track_fetch.py and
provide the output of that script as an input argument for this one.

IMPORTANT: If using this script elsewhere than on Boden then rememer to use the option --root-dir to
set the directory in which the 7Digital mp3 files are stored.


Functions
---------
- set_path_h5
    Set path to the msd_summary_file.h5.

- set_path_txt_mismatches
    Set path to the msd_mismatches.txt file.

- set_path_txt_duplicates
    Set path to the msd_duplicates.txt file.

- extract_ids_from_summary
    Produce a dataframe with 7digitalid's and tid's of all tracks in the dataset.

- df_merge
    Produce a dataframe with 7digitalid's, tid's and paths of mp3 files on the server.

- df_purge_mismatches
    Remove mismatches from previously generated dataset.

- df_purge_faulty_mp3_1
    Remove tracks which have file size 0.

- df_purge_faulty_mp3_2
    Remove tracks which can't be opened and therefore have NaN length.

- df_purge_no_tag
    Remove tracks which are not matched to any tag.

- read_duplicates
    Read the msd_duplicates.txt file and produces a list (of lists) of duplicates.

- df_purge_duplicates
    Retain only one track for each set of duplicates.

- ultimate_output
    Combine all the previous functions and produces a dataframe accoring to the given parameters.
'''

import argparse
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
from itertools import islice

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../modules')))

import query_lastfm as db

path_h5 = '/srv/data/msd/msd_summary_file.h5'
path_txt_mismatches = '/srv/data/msd/sid_mismatches.txt'
path_txt_duplicates = '/srv/data/urop/msd_duplicates.txt'

def set_path_h5(new_path):
    ''' Set path to the msd_summary_file.h5. '''

    global path_h5
    path_h5 = new_path

def set_path_txt_mismatches(new_path):
    ''' Set path to the msd_mismatches.txt. '''

    global path_txt_mismatches
    path_txt_mismatches = new_path

def set_path_txt_duplicates(new_path):
    ''' Set path to the msd_duplicates.txt. '''

    global path_txt_duplicates
    path_txt_duplicates = new_path

def extract_ids_from_summary(): 
    ''' Produce a dataframe with 7digitalid's and tid's of all tracks in the dataset. '''

    with h5py.File(path_h5, 'r') as h5:
        dataset_1 = h5['metadata']['songs']
        dataset_2 = h5['analysis']['songs']
        dataset_3 = h5['musicbrainz']['songs']
        df_summary = pd.DataFrame(data={'track_7digitalid': dataset_1['track_7digitalid'], 'track_id': dataset_2['track_id'], 'year': dataset_3['year']})
        df_summary['track_id'] = df_summary['track_id'].apply(lambda x: x.decode('UTF-8'))
        df_summary['year'] = df_summary['year'].fillna(0)
        return df_summary

def df_merge(track_summary_df: pd.DataFrame, merged_df: pd.DataFrame):
    ''' Produce a dataframe with 7digitalid's, tid's and paths of mp3 files on the server. '''


    df = pd.merge(track_summary_df, merged_df, on='track_7digitalid', how='inner')
    df = df[-df.duplicated('track_7digitalid', keep=False)]
    df = df[-df.duplicated('track_id', keep=False)]
    return df

def df_purge_mismatches(merged_df: pd.DataFrame):
    ''' Remove mismatches from previously generated dataset. '''

    df = merged_df.set_index('track_id')
    to_drop = []
    with open(path_txt_mismatches, encoding='utf-8') as file: 
        for line in file:
            to_drop.append(re.findall(r'T[A-Z0-9]{17}', line)[0])
    to_drop = [tid for tid in to_drop if tid in df.index]
    df.drop(to_drop, inplace=True)
    return df.reset_index()

def df_purge_faulty_mp3_1(merged_df: pd.DataFrame):
    ''' Remove tracks which have file size 0. '''

    df = merged_df[merged_df['file_size'] > 0]
    return df

def df_purge_faulty_mp3_2(merged_df: pd.DataFrame):
    ''' Remove tracks which can't be opened and therefore have NaN length. '''

    df = merged_df[-merged_df['clip_length'].isna()]
    return df

def df_purge_no_tag(merged_df: pd.DataFrame, path_db: str = None):
    ''' Remove tracks which are not matched to any tag. '''

    if path_db:
        db.set_path(path_db)

    lastfm = db.LastFm(db.path)

    tids_with_tag = lastfm.get_tids_with_tag()
    tids_with_tag_df = pd.DataFrame(data={'track_id': tids_with_tag})
    
    return pd.merge(merged_df, tids_with_tag_df, on='track_id', how='inner')

def read_duplicates():      
    ''' Read the msd_duplicates.txt file and produces a list (of lists) of duplicates. '''

    l = []
    with open (path_txt_duplicates, encoding='utf-8') as file:
        t = []
        for line in islice(file, 7, None):
            if line[0] == '%':
                l.append(t)
                t = []
            else:
                t.append(re.findall(r'T[A-Z0-9]{17}', line)[0])
        l.append(t)
    return l

def df_purge_duplicates(merged_df: pd.DataFrame, randomness: bool = False):
    ''' Retain only one track for each set of duplicates. '''

    df = merged_df.set_index('track_id')
    dups = read_duplicates()
    idxs = df.index
    to_drop = [[tid for tid in sublist if tid in idxs] for sublist in dups] # contains lists of duplicates which are also in the purged dataframe
    if not randomness:
        np.random.seed(42)
    for sublist in to_drop:
        if len(sublist) > 1: # for each list of duplicates, only "save" one track (that is, pop it from to_drop)
            sublist.pop(np.random.randint(len(sublist)))
        else:
            continue
    to_drop = [tid for sublist in to_drop for tid in sublist] # flatten
    df.drop(to_drop, inplace=True)
    return df.reset_index()

def ultimate_output(df: pd.DataFrame, discard_no_tag: bool = False, discard_dupl: bool = False):
    ''' Produces a dataframe with the following columns: 'track_id', 'track_7digitalid', 'file_path', 'file_size', 'channels', 'clip_length'.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to purge.
    
    discard_no_tag : bool
        If True, discards tracks which are not matched to any tag.

    discard_dupl : bool
        If True, discards tracks which are duplicates and keeps one for each set.

    Returns
    -------
    df : pd.DataFrame
        The columns are the ones listed above.
        The entries are the ones specified by the given parameters.
    '''

    print("Fetching mp3 files from input dataframe...", end=" ", flush=True)
    merged_df = df_merge(extract_ids_from_summary(), df)
    print("done")

    print("Purging mismatches...", end=" ", flush=True)
    merged_df = df_purge_mismatches(merged_df)
    print("done")

    print("Purging faulty mp3 files...")
    print("    Checking mp3 files which have size 0...", end=" ", flush=True)
    merged_df = df_purge_faulty_mp3_1(merged_df)
    print("done")
    print("    Checking mp3 files which can't be opened and have length 0...", end=" ", flush=True)
    merged_df = df_purge_faulty_mp3_2(merged_df)
    print("done")
    
    if discard_no_tag:
        print("Purging tracks with no tags...", end=" ", flush=True)
        merged_df = df_purge_no_tag(merged_df)
        print("done")
    
    if discard_dupl:
        print("Purging duplicate tracks...", end=" ", flush=True)
        merged_df = df_purge_duplicates(merged_df)
        print("done")
    
    return merged_df

if __name__ == "__main__":
    
    description = "Script to merge the list of mp3 files obtained with track_fetch.py with the MSD summary file, remove unwanted entries such as mismatches, faulty files or duplicates, and output a csv file with the following columns: 'track_id', 'track_7digitalid', 'file_path', 'file_size', 'channels', 'clip_length'."
    epilog = "Example: python track_wrangle.py /data/track_on_boden.csv ./wrangl.csv --discard-no-tag"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("input", help="input csv filename or path")
    parser.add_argument("output", help="output csv filename or path")
    parser.add_argument("--path-h5", help="set path to msd_summary_file.h5")
    parser.add_argument("--path-txt-dupl", help="set path to duplicates info file")
    parser.add_argument("--path-txt-mism", help="set path to mismatches info file")
    parser.add_argument("--discard-no-tag", action="store_true", help="choose to discard tracks with no tags")
    parser.add_argument("--discard-dupl", action="store_true", help="choose to discard duplicate tracks")
    
    args = parser.parse_args()
    
    if args.output[-4:] != '.csv':
        output = args.output + '.csv'
    else:
        output = args.output

    if os.path.isfile(output):
       print("WARNING file " + output + " already exists!")
       sys.exit(0)

    if args.path_h5:
        path_h5 = os.path.expanduser(args.path_h5)
    if args.path_txt_mism:
        path_txt_mismatches = os.path.expanduser(args.path_txt_mism)
    if args.path_txt_dupl:
        path_txt_duplicates = os.path.expanduser(args.path_txt_dupl)
    
    df = pd.read_csv(args.input, comment='#')

    assert 'file_size' in df and 'clip_length' in df

    df = ultimate_output(df, args.discard_no_tag, args.discard_dupl)
    
    # create output csv file
    with open(output, 'a') as f:
        # insert comment line displaying options used
        comment = '# python'
        comment += ' ' + os.path.basename(sys.argv.pop(0))
        
        options = [arg for arg in sys.argv if arg not in (args.input, args.output)]
        for option in options:
            comment += ' ' + option
        
        comment += ' ' + os.path.basename(args.input) + ' ' + os.path.basename(output)
        
        # write comment to the top line
        f.write(comment + '\n')
        
        # write dataframe
        df.to_csv(f, index=False)
