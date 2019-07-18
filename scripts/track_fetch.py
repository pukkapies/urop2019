''' Contains tools to fetch MP3 files on our server and analyze their size, length and number of channels


Notes
-----
This file can be run as a script. To do so, just type 'python track_fetch.py' in the terminal. The help 
page should contain all the options you might possibly need.

IMPORTANT: If using this script elsewhere than on Boden then rememer to use the option --root-dir to
set the directory in which the 7Digital MP3 files are stored.


Functions
---------
- set_mp3_root_dir              Tells the script where MP3 files are stored
- find_tracks                   Performs an os.walk to find all the MP3 files within mp3_root_dir
- find_tracks_with_7dids        Extracts the 7Digital ID from the MP3 filenames
- check_size                    Extends the columns of the given dataframe to identify the size of the tracks
- check_mutagen_info            Extends the columns of the given dataframe to identify if the tracks 
                                can be opened and the duration and number of channels of the tracks
'''

import mutagen.mp3
import os
import pandas as pd
import sys

mp3_root_dir = '/srv/data/msd/7digital/'

def set_mp3_root_dir(new_root_dir): # DAVIDE: now same function name and var name across all modules, to avoid errors
    global mp3_root_dir
    mp3_root_dir = new_root_dir

def find_tracks():
    paths = []
    for folder, subfolders, files in os.walk(mp3_root_dir):
        for file in files:
            path = os.path.join(os.path.abspath(folder), file)
            paths.append(path)
    paths = [path for path in paths if path[-4:] == '.mp3']
    return paths

def find_tracks_with_7dids():
    paths = find_tracks()
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
            The file size of the MP3 file.
        
    
    '''
    s = []
    for path in df['paths']: 
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
            continue
        
        if verbose == True:
            if idx % 1000 == 0:
                print('PROGRESS: {:6d}/{:6d}'.format(idx, tot))

    if add_length == True: 
        #df['length'] = pd.Series(l, index=df.index)
        #df['lengths'] = pd.Series(l, index=df.index) # ADEN: 'length' is better since df.length is ambiguous...
        df['track_length'] = pd.Series(l, index=df.index) # DAVIDE: it is a column name, I'm not happy with plural. 'track_length'? 
    if add_channels == True:
        df['channels'] = pd.Series(c, index=df.index) # DAVIDE: 'channels' though must necessarily be plural, since 'channel' makes no sense
    return df

def die_with_usage():
    print()
    print("track_fetch.py - Script to search for MP3 files within mp3_root_dir and output a CSV file with (optionally) the")
    print("                 following columns: 'path', 'track_7digitalID', 'track_length, 'file_size', 'channels'")
    print()
    print("Usage:     python track_fetch.py <output filename> [options]")
    print()
    print("General Options:")
    print("  --no-size              Do not add column containing file sizes to output file.")
    print("  --no-length            Do not add column containing track lengths to output file.")
    print("  --no-channels          Do not add column containing track number of channels to output file.")
    print("  --root-dir             Set different mp3_root_dir.")
    print("  --help                 Show this help message and exit.")
    print("  --verbose              Show progress.")
    print()
    print("Example:   python track_fetch.py /data/tracks_on_boden.csv --root-dir /data/songs/ --no-channels --verbose")
    print()
    sys.exit(0)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        die_with_usage()
    
    if '--help' in sys.argv:
        if len(sys.argv) == 2:
            die_with_usage()
        else:
            print("???")
            sys.exit(0)
    
    if sys.argv[1][-4:] == '.csv':
        output = sys.argv[1]
    else:
        output = sys.argv[1] + '.csv'

    add_size = True
    add_length = True
    add_channels = True
    verbose = False

    while True:
        if len(sys.argv) == 2:
            break
        elif sys.argv[2] == '--root-dir':
            set_mp3_root_dir(sys.argv[3])
            del sys.argv[2:4]
        elif sys.argv[2] == '--no-size':
            add_size = False
            del sys.argv[2]
        elif sys.argv[2] == '--no-length':
            add_length = False
            del sys.argv[2]   
        elif sys.argv[2] == '--no-channels':
            add_channels = False
            del sys.argv[2]
        elif sys.argv[2] == '--verbose':
            verbose = True
            del sys.argv[2]     
        else:
            print("???")
            sys.exit(0)

        df = find_tracks_with_7dids()
        if add_length == True or add_channels == True:
            df = check_mutagen_info(df, add_length, add_channels, verbose)
        if add_size == True:
            df = check_size(df)
        df.to_csv(output, index=False)