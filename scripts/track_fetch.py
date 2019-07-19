''' Contains tools to fetch mp3 files on our server and analyze their size, length and number of channels


Notes
-----
This file can be run as a script. To do so, just type 'python track_fetch.py' in the terminal. The help 
page should contain all the options you might possibly need.

IMPORTANT: If using this script elsewhere than on Boden then rememer to use the option --root-dir to
set the directory in which the 7Digital mp3 files are stored.


Functions
---------
- set_mp3_root_dir
    Tell the script where mp3 files are stored.

- find_tracks
    Performs an os.walk to find all the mp3 files within mp3_root_dir

- find_tracks_with_7dids
    Extract 7digitalid's from mp3 filenames.

- check_size
    Extend the columns of the given dataframe to identify the size of the tracks.

- check_mutagen_info
    Extend the columns of the given dataframe to identify whether the tracks can be
    opened and the duration and number of channels of the tracks.
'''

import os
import sys
import argparse
import time

import pandas as pd
from mutagen.mp3 import MP3
from mutagen.mp3 import HeaderNotFoundError

mp3_root_dir = '/srv/data/msd/7digital/'

def set_mp3_root_dir(new_root_dir): # DAVIDE: now same function name and var name across all modules, to avoid errors
    """ Function to set mp3_root_dir, useful when script is used as module """

    global mp3_root_dir
    mp3_root_dir = new_root_dir

def find_tracks(abs_path = False):
    paths = []
    for folder, subfolders, files in os.walk(mp3_root_dir):
        for file in files:
            path = os.path.join(folder, file)
            paths.append(path)
    paths = [path for path in paths if path[-4:] == '.mp3']
    if abs_path == False:
        paths = [os.path.relpath(path, mp3_root_dir) for path in paths]
    return paths

def find_tracks_with_7dids(abs_path = False):
    paths = find_tracks(abs_path)
    paths_7dids = [int(os.path.basename(path)[:-9]) for path in paths]
    df = pd.DataFrame(data={'path': paths, 'track_7digitalid': paths_7dids})
    return df

def check_size(df):
    '''
    Parameters
    ----------
    df: pd.DataFrame
        The input dataframe which you want extra information (length and number 
        of channel of tracks).
        
    Returns
    -------
    df: pd.DataFrame
        A dataframe that has one extra column:
        
        'file_size': float
            The file size of the mp3 file.
        
    
    '''
    s = []
    for path in df['path']: 
        # path = mp3_root_dir[:-1] + path # DAVIDE: what's wrong with os.path.join? string concatenation is more dangerous, what if path is an absolute path?
        path = os.path.join(mp3_root_dir, path)
        s.append(os.path.getsize(path))
    #df['size'] = pd.Series(s, index=df.index) # ADEN: 'sizes' is better since df.size is ambiguous...
    #df['sizes'] = pd.Series(s, index=df.index) # DAVIDE: it is a column name, I'm not happy with plural. 'file_size'? 
    df['file_size'] = pd.Series(s, index=df.index)
    return df

def check_mutagen_info(df, verbose = False, debug: int = None):
    '''
    Parameters
    ----------
    df: pd.DataFrame
        The input dataframe which you want extra information (length and number 
        of channel of tracks).

    Returns
    -------
    df: pd.DataFrame
        A dataframe that has two extra columns if add_length and add_channels are True:
        
        'length': float
            The duration in seconds of the mp3 tracks.
            
        NOTE: an empty cell is returned to the corresponding rows for 'track_length' 
        and 'channels' if the script cannot read the size of the tracks or cannot 
        open the tracks (i.e. broken tracks).
    '''
    start = time.time()
    l = []
    tot = len(df)
    for idx, path in enumerate(df['path']):
        path = os.path.join(mp3_root_dir, path)
        try:
            l.append(MP3(path).info.length)
        except HeaderNotFoundError:
            l.append('')
        except:
            print('WARNING unknown exception occurred at {:6d}, {}'.format(idx, path))
        
        if verbose == True:
            if idx % 5000 == 0:
                print('Processed {:6d} in {:8.4f} sec. Progress: {:2d}%'.format(idx, time.time() - start, int(idx / tot * 100)))

        if debug:
            if idx == debug:
                return l

    df['length'] = pd.Series(l, index=df.index) # 'lengths'? 'track_length'?

    if verbose == True:
        print('Processed {:6d} in {:8.4f} sec.'.format(tot, time.time() - start))
    
    return df

if __name__ == "__main__":

    description = """Script to search for mp3 files within mp3_root_dir and output a csv file with (optionally) the 
                     following columns: 'path', 'track_7digitalID', 'track_length, 'file_size', 'channels'"""
    epilog = "Example:   python track_fetch.py /data/tracks_on_boden.csv --root-dir /data/songs/ --no-channels --verbose"

    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("output", help="Output filename")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show progress.")
    parser.add_argument("--root-dir", help="Set different mp3_root_dir.")
    parser.add_argument("--abs", action="store_true", dest="abs_path", help="Use absolute paths, instead of relative, in output file.")
    parser.add_argument("--skip-os", action="store_false", dest="use_os", help="Do not calculate tracks size.")
    parser.add_argument("--skip-mutagen", action="store_false", dest="use_mutagen", help="Do not use mutagen to check tracks length.")
    parser.add_argument("--debug", help="Enable debug mode")

    args = parser.parse_args()
   
    if args.output[-4:] != '.csv':
        args.output = args.output + '.csv' 
    if args.root_dir:
        mp3_root_dir = args.root_dir    
    
    df = find_tracks_with_7dids(args.abs_path)

    if args.debug == True:
        import pdb
        pdb.set_trace()
        sys.exit(0)

    if args.use_os == True:
        df = check_size(df)

    if args.use_mutagen == True:
        df = check_mutagen_info(df, args.verbose)
    
    df.to_csv(args.output, index=False)
