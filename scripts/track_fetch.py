''' Contains tools to fetch mp3 files on our server and analyze their size, length and number of channels


Notes
-----
This file can be run as a script. To do so, just type 'python track_fetch.py' in the terminal. The help 
page should contain all the options you might possibly need.

IMPORTANT: If using this script elsewhere than on Boden then rememer to use the option --root-dir to
set the directory in which the 7Digital mp3 files are stored.


Functions
---------
- find_tracks                   Performs an os.walk to find all the mp3 files within mp3_root_dir

- find_tracks_with_7dids        Extracts the 7Digital ID from the mp3 filenames

- check_size                    Extends the columns of the given dataframe to identify the size of the tracks

- check_mutagen_info            Extends the columns of the given dataframe to identify whether the tracks can be
                                opened and the duration and number of channels of the tracks
'''

import os
import sys
import argparse

import mutagen.mp3
import pandas as pd

mp3_root_dir = '/srv/data/msd/7digital/'

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

# ADEN: def check_mutagen_info(df, add_length=True, add_channels=True, verbose=True, save_csv=True, output_path='/srv/data/urop/ultimate_csv_size.csv'):
def check_mutagen_info(df, add_length=True, add_channels=True, verbose=True): # DAVIDE: check out 'if __name__ = __main__'; this script outputs a csv, there's no need to mention csv's in function declarations
    '''
    Parameters
    ----------
    df: pd.DataFrame
        The input dataframe which you want extra information (length and number 
        of channel of tracks).
    
    add_length: bool
        If True, the computed lengths column is appended to the df.
    
    add_channels: bool
        If True, the computed number of channels column is appended to the df.
        
    verbose: bool
        If True, the progress of running the program is printed.
        
        
    Returns
    -------
    df: pd.DataFrame
        A dataframe that has two extra columns if add_length and add_channels are True:
        
        'track_length': float
            The duration of the mp3 tracks.
            
        'channels': float
            The number of channels of the mp3 tracks.
            
        NOTE: an empty cell is returned to the corresponding rows for 'track_length' 
        and 'channels' if the script cannot read the size of the tracks or cannot 
        open the tracks (i.e. broken tracks).
    '''
    
    tot = len(df)
    l = []
    c = []
    for idx, path in enumerate(df['path']):
        path = os.path.join(mp3_root_dir, path)
        # path = mp3_root_dir[:-1]+ path # DAVIDE: same comment as above...
        try:
            audio = mutagen.mp3.MP3(path)
            l.append(audio.info.length)
            c.append(audio.info.channels)
        except:
            l.append('')
            c.append('')
            print('ERROR opening ' + path)
            continue
        
        if verbose == True:
            if idx % 1000 == 0:
                print('Processed {:6d} out of {:6d}...'.format(idx, tot))
    
    if verbose == True:
        print('Processed {:6d} out of {:6d}...'.format(tot, tot))


    if add_length == True: 
        #df['length'] = pd.Series(l, index=df.index)
        #df['lengths'] = pd.Series(l, index=df.index) # ADEN: 'length' is better since df.length is ambiguous...
        df['track_length'] = pd.Series(l, index=df.index) # DAVIDE: it is a column name, I'm not happy with plural. 'track_length'? 
    if add_channels == True:
        df['channels'] = pd.Series(c, index=df.index) # DAVIDE: 'channels' though must necessarily be plural, since 'channel' makes no sense
    return df

if __name__ == "__main__":

    description = """Script to search for mp3 files within mp3_root_dir and output a csv file with (optionally) the 
                     following columns: 'path', 'track_7digitalID', 'track_length, 'file_size', 'channels'"""
    epilog = "Example:   python track_fetch.py /data/tracks_on_boden.csv --root-dir /data/songs/ --no-channels --verbose"

    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("output", help="Output filename")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show progress.")
    parser.add_argument("--root-dir", help="Set different mp3_root_dir.")
    parser.add_argument("--abs-path", action="store_true", help="Use absolute paths in output file.")
    parser.add_argument("--no-size", action="store_true", help="Do not add column containing file sizes to output file.")
    parser.add_argument("--no-length",action="store_true", help="Do not add column containing track lengths to output file.")
    parser.add_argument("--no-channels", action="store_true", help="Do not add column containing track number of channels to output file.")

    args = parser.parse_args()
   
    if args.output[-4:] != '.csv':
        args.output = args.output + '.csv' 
    if args.root_dir:
        mp3_root_dir = args.root_dir    
    
    add_length = not args.no_length 
    add_channels = not args.no_channels


    df = find_tracks_with_7dids(args.abs_path)
    if add_length == True or add_channels == True:
        df = check_mutagen_info(df, add_length, add_channels, args.verbose)
    if args.no_size != True:
        df = check_size(df)
    df.to_csv(args.output, index=False)
