''' Contains tools for fetching MP3 files on the server, matching 7digitalid's with tid's, and purging unwanted entries such as mismatches, faulty MP3 files, tracks without tags or duplicates

Notes
-----
The purpose of this module is to provide an ultimate_output() function returning a
dataframe with three columns: track_id, track_7digitalid and path.

The only function which does not contribute to the above-mentioned purpose is 
read_duplicates_and_purge(), which returns a list of sublists where each sublists contains the
duplicate tracks for a particular songs, discarding all the tracks which are damaged or not
on our server.

When running the ultimate_output() function, one can choose whether or not to exclude tracks with
no tags, whether or not to exclude duplicates, and what the minimum size (in bytes) of the MP3
files should be. With all the parameters left to their default value, the function takes slightly
less than a minute to run on our server. Nevertheless, in order not to have to run the same 
function every time such a dataset is needed, a CLI script mp3_wrangler.py is provided, which 
relies on the functions contained in this modules and produces a CSV file. 


Functions
---------
- set_path_h5                Sets path to the msd_summary_file.h5
- set_path_txt_mismatches    Sets path to the msd_mismatches.txt file
- set_path_txt_duplicates    Sets path to the msd_duplicates.txt file
- extract_ids_from_summary   Produces a dataframe with 7digitalid's and tid's of all tracks in the dataset
- find_tracks                Gets a list of paths of MP3 files on our server
- find_tracks_with_7dids     Produces a dataframe with 7digitalid's and paths of MP3 files on our server
- df_merge                   Produces a dataframe wiht 7digitalid's, tid's and paths of MP3 files on our server
- df_purge_mismatches        Removes mismatches from previously generated dataset
- get_idx_mp3_size_zero      Gets a list of indexes in the dataset of tracks whose size is 0
- get_idx_mp3_size_less_than Gets a list of indexes in the dataset of tracks whose size is less than a threshold
- df_purge_faulty_mp3        Removes tracks whose size is less than a certain threshold
- df_purge_no_tag            Removes tracks which are not matched to any tag
- read_duplicates            Reads the msd_duplicates.txt file and produces a list (of lists) of duplicates
- read_duplicates_and_purge  Reads the msd_duplicates.txt file and keeps only "good" tracks
- df_purge_duplicates        Retains only one track for each set of duplicates
- ultimate_output            Combine the previous functions and produces a dataframe accoring to the given parameters
'''

import h5py
import os
import pandas as pd
import sys

from itertools import islice

if os.path.basename(os.getcwd()) == 'scripts':
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../modules')))
else:
    sys.path.insert(0, os.path.join(os.getcwd(), 'modules'))

import lastfm_query as db

MP3_ROOT_DIR = '/srv/data/msd/7digital/'

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


### functions to fetch MP3 files on server and remove mismatches

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

def df_purge_mismatches(track_df: pd.DataFrame):
    df = track_df.set_index('track_id')
    to_drop = []
    with open(path_txt_mismatches, 'r') as file:
        for line in file:
            to_drop.append(line[27:45])
    to_drop = [tid for tid in to_drop if tid in df.index]
    df.drop(to_drop, inplace=True)
    return df.reset_index()


### functions to find tracks with too small a file size and purge them

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


### functions to find tracks with no tags and purge them
    
def df_purge_no_tag(track_df: pd.DataFrame, lastfm_db: str = None):
    if lastfm_db:
        db.set_path(lastfm_db)

    tids_with_tag = db.get_tids_with_tag()
    tids_with_tag_df = pd.DataFrame(data={'track_id': tids_with_tag})
    
    return pd.merge(track_df, tids_with_tag_df, on='track_id', how='inner')


### functions to tackle duplicates

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

def df_purge_duplicates(track_df: pd.DataFrame, mode: str = 'random'):
    dups = read_duplicates()
    idxs = track_df.set_index('track_id').index
    dups_purged = [[tid for tid in sublist if tid in idxs] for sublist in dups]

    if mode == 'random': # the only mode currently supported
        df = track_df.set_index('track_id')
        to_drop = dups_purged
        for subset in to_drop:
            if len(subset) > 1:
                subset.pop()
            else:
                continue
        to_drop = [tid for sublist in to_drop for tid in sublist]
        df.drop(to_drop, inplace=True)
        return df.reset_index()
    else:
        raise NameError("mode '" + mode + "' is not defined")


### output functions

def ultimate_output(threshold: int = 0, discard_no_tag: bool = False, discard_dupl: bool = False):
    ''' Produces a dataframe with the following columns: 'track_id', 'track_7digitalid' and 'path'.
    
    Parameters
    ----------
    threshold : int
        the tracks with file size (in bytes) below the threshold are discarded (50000 is a safe value)
    
    discard_no_tag : bool
        if True, discards tracks which are not matched to any tag

    discard_dupl : bool
        if True, discards tracks which are duplicates and keeps one for each set

    Returns
    -------
    df : pd.DataFrame
        - columns are 'track_id', 'track_7digitalid' and 'path'
        - entries are all the tracks on our server which are not mismatched and satisfy the given parameters
    '''

    print('Fetching mp3 files from root directory...', end=' ')
    df = df_merge(extract_ids_from_summary(), find_tracks_with_7dids())
    print('done')

    print('Purging mismatches...', end=' ')
    df = df_purge_mismatches(df)
    print('done')

    print('Purging faulty MP3 files...', end=' ')
    df = df_purge_faulty_mp3(df, threshold=threshold)
    print('done')
    
    if discard_no_tag == True:
        print('Purging tracks with no tags...', end=' ')
        df = df_purge_no_tag(df)
        print('done')
    
    if discard_dupl == True:
        print('Purging duplicate tracks...', end=' ')
        df = df_purge_duplicates(df)
        print('done')

#     if add_length == True:
#         print('Checking length of audio tracks...', end=' ')
#         lengths = []
#         for path in df['path']:
#             mp3 = mutagen.mp3.MP3(os.path.join(MP3_ROOT_DIR, path))
#             lengths.append(mp3.info.length)
#         df['track_length'] = pd.Series(lenghts, index=df.index)
#         print('done')
    
    return df

def read_duplicates_and_purge(threshold: int = 0, discard_no_tag: bool = False):
    ''' Produces a list containing ONLY tracks on our server where each sublists contains all the duplicates for a song
    
    Parameters
    ----------
    threshold : int
        the tracks with file size (in bytes) below the threshold are discarded (50000 is a safe value)
    
    discard_no_tag : bool
        if True, discards tracks which are not matched to any tag

    Returns
    -------
    dups_purged : list of lists
        - elements are all the 53471 sets of duplicates as given in the msd_duplicates.txt file (some might now be empty)
        - elements within lists are all the duplicate tids of some specific song - discarding inexistent tracks, mismatches and faulty audio files 
    '''
    df = df_merge(extract_ids_from_summary(), find_tracks_with_7dids())
    df = df_purge_mismatches(df)
    df = df_purge_faulty_mp3(df, threshold=threshold)
    
    if discard_no_tag == True:
        df = df_purge_no_tag(df)

    dups = read_duplicates()
    idxs = df.set_index('track_id').index
    dups_purged = [[tid for tid in sublist if tid in idxs] for sublist in dups]
    return dups_purged

# script

def die_with_usage():
    print()
    print("Mp3Wrangler - Script to fetch Mp3 songs from the server and output an ultimate CSV file")
    print()
    print("Usage:     python data_wrangle.py <csv filename or path> [options]")
    print()
    print("General Options:")
    print("  --discard-no-tag       Choose to discard tracks with no tags.")
    print("  --discard-dupl <mode>  Choose to discard duplicate tracks.") # <mode> not currently supported
    print("  --help                 Show this help message and exit.")
    print("  --threshold            Set the minimum size (in bytes) to allow for the MP3 files (default 0).")
    print()
    print("Example:   python data_wrangle.py ./wrangl.csv --threshold 50000 --discard-no-tag")
    print()
    sys.exit(0)

if __name__ == "__main__":

    # show help
    if len(sys.argv) < 2:
        die_with_usage()
    
    # show help, if user did not input something weird
    if '--help' in sys.argv:
        if len(sys.argv) == 2:
            die_with_usage()
        else:
            print('???')
            sys.exit(0)
    
    if sys.argv[1][-4:] == '.csv':
        output = sys.argv[1]
    else:
        output = sys.argv[1] + '.csv'

    # check arguments
    if len(sys.argv) == 2:
        df = ultimate_output()
        df.to_csv(output, index=False)
    else:
        # initialize variables
        threshold = 0 
        discard_no_tag = False
        discard_dupl = False

        while True:
            if len(sys.argv) == 2:
                break
            elif sys.argv[2] == '--threshold':
                threshold = int(sys.argv[3])
                del sys.argv[2:4]
            elif sys.argv[2] == '--discard-no-tag':
                discard_no_tag = True
                del sys.argv[2]
            elif sys.argv[2] == '--discard-dupl':
                discard_dupl = True
                del sys.argv[2]      
            else:
                print('???')
                sys.exit(0)

        df = ultimate_output(threshold, discard_no_tag, discard_dupl)
        df.to_csv(output, index=False)