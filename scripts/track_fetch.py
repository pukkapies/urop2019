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
    Performs an os.walk to find all the mp3 files within mp3_root_dir.

- find_tracks_with_7dids
    Finds all mp3 files within mp3_root dir and extracts the 7digitalid's from mp3 filenames.

- check_size
    Extend the columns of the given dataframe to identify the size of the tracks.

- check_mutagen_info
    Extend the columns of the given dataframe to identify whether the tracks can be
    opened and the duration and number of channels of the tracks.
'''

import argparse
import os
import sys
import time

import pandas as pd
from mutagen.mp3 import MP3
from mutagen.mp3 import HeaderNotFoundError

mp3_root_dir = '/srv/data/msd/7digital/'

def set_mp3_root_dir(new_root_dir): 
    ''' Function to set mp3_root_dir, useful when script is used as module. '''

    global mp3_root_dir
    mp3_root_dir = new_root_dir

def find_tracks(abs_path = False):
    ''' Finds all the mp3 files within mp3_root_dir. 
    
    Parameters
    ----------
    abs_path : bool
        If true, use absolute paths.

    Returns
    -------
    paths : list
        list containing paths (str) to all mp3 files in mp3_root_dir.
    
    '''

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
    ''' Finds all the mp3 files within mp3_root_dir and extracts 7digitalid from their names
    
    Parameters
    ----------
    abs_path : bool
        If true, use absolute paths.

    Returns
    -------
    df : DataFrame
        contains 1 column for the paths and 1 for the corresponding 7digitalids.
    '''

    paths = find_tracks(abs_path)
    paths_7dids = [int(os.path.basename(path)[:-9]) for path in paths]
    df = pd.DataFrame(data={'track_7digitalid': paths_7dids, 'file_path': paths})
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
    for path in df['file_path']: 
        path = os.path.join(mp3_root_dir, path)
        s.append(os.path.getsize(path))
    df['file_size'] = pd.Series(s, index=df.index)
    return df

def check_mutagen_info(df, verbose = True, debug: int = None):
    '''
    Parameters
    ----------
    df: pd.DataFrame
        The input dataframe which you want extra information (length and number 
        of channel of tracks).
        
    verbose: bool
        If True, print progress.
        
    debug: int
        If not None, return partial list when idx == debug.

    Returns
    -------
    df: pd.DataFrame
        A dataframe that has two extra columns if add_length and add_channels are True:
        
        'length': float
            The duration in seconds of the mp3 tracks.
            
        NOTE: an empty cell is returned to the corresponding rows for 'clip_length' 
        and 'channels' if the script cannot read the size of the tracks or cannot 
        open the tracks (i.e. broken tracks).
    '''

    start = time.time()
    l = []
    c = []
    tot = len(df)
    for idx, path in enumerate(df['file_path']):
        path = os.path.join(mp3_root_dir, path)
        try:
            l.append(MP3(path).info.length)
            c.append(MP3(path).info.channels)
        except HeaderNotFoundError:
            l.append('')
            c.append('')
        
        if verbose == True:
            if idx % 500 == 0:
                print('Processed {:6d} in {:8.4f} s. Progress: {:2d}%'.format(idx, time.time() - start, int(idx / tot * 100)))

        # debug option allows to stop execution after 'debug' steps
        if debug:
            if idx == debug:
                return l # returning l or returning c is analogous for debugging purposes

    df['channels'] = pd.Series(c, index=df.index)
    df['channels'] = df['channels'].fillna(0) # by default missing values are NaN (which pandas considers as float)
    df['channels'] = df['channels'].apply(lambda x: int(x))
    df['clip_length'] = pd.Series(l, index=df.index) 

    if verbose == True:
        print('Processed {:6d} in {:8.4f} s.'.format(tot, time.time() - start))
    
    return df

if __name__ == "__main__":
    
    description = "Script to search for mp3 files within mp3_root_dir and output a csv file with (optionally) the following columns: 'track_7digitalID', 'file_path', 'file_size', 'channels', 'clip_length'."
    epilog = "Example: python track_fetch.py /data/tracks_on_boden.csv --root-dir /data/songs/ --verbose"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("output", help="output filename or path")
    parser.add_argument("-v", "--verbose", action="store_true", help="show progress")
    parser.add_argument("--root-dir", help="set directory to find mp3 files")
    parser.add_argument("--abs", action="store_true", dest="abs_path", help="use absolute paths in output file")
    parser.add_argument("--no-size", action="store_false", dest="use_os", help="do not calculate tracks size")
    parser.add_argument("--no-mut-length", action="store_false", dest="mutg_length", help="do not use mutagen to check tracks number of channels")
    parser.add_argument("--no-mut-channels", action="store_false", dest="mutg_channels", help="do not use mutagen to check tracks length")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")

    args = parser.parse_args()
   
    if args.output[-4:] != '.csv':
        output = args.output + '.csv'
    else:
        output = args.output
    
    if os.path.isfile(args.output):
       print("WARNING file " + args.output + " already exists!")
       sys.exit(0)
    
    if args.root_dir:
        mp3_root_dir = os.path.expanduser(args.root_dir)
    
    df = find_tracks_with_7dids(args.abs_path)

    if args.debug == True:
        import pdb
        pdb.set_trace()
        sys.exit(0)

    if args.use_os == True:
        df = check_size(df)

    if args.mutg_length == True and mutg_channels == True:
        df = check_mutagen_info(df, args.verbose)
        if args.mutg_length == False:
            df.drop('clip_length', axis=1, inplace=True)
        if args.mutg_channels == False:
            df.drop('channels', axis=1, inplace=True)
    
    with open(output, 'a') as f:
        # insert comment line displaying optiones used
        comment = '# python'
        comment += ' ' + os.path.basename(sys.argv.pop(0))
        options = [arg for arg in sys.argv if arg != args.output]
        for option in options:
            comment += ' ' + option
        comment += ' ' + os.path.basename(output)
        f.write(comment + '\n')
        df.to_csv(f, index=False)
