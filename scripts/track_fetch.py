"""

"""
'''
Note
----

Functions
---------
- set_mp3_root_dir
    Tell the script the root directory of where mp3s were stored.
    
- extract_ids_from_summary

- find_tracks

- find_tracks_with_7dids

- check_size
    Extend the column of the given dataframe to identify sizes of tracks.

- check_mutagen_info
    Extend the columns of the given dataframe to identify if a track can be 
    opened and the duration and number of channels of a track.
    


'''
import h5py
import mutagen.mp3
import os
import pandas as pd
import sys

root_dir = '/srv/data/msd/7digital/'

def set_mp3_root_dir(new_root_dir):   #better change it to another name, or will mess up with mp3_to_mpz when using from xx import *, please change
    '''
    Parameters
    ----------
    
    new_path: str
        The root directory of where mp3s were stored.
        
    '''
    global root_dir
    root_dir = new_root_dir

def extract_ids_from_summary(path = '/srv/data/msd/msd_summary_file.h5'):
    with h5py.File(path, 'r') as h5:
        dataset_1 = h5['metadata']['songs']
        dataset_2 = h5['analysis']['songs']
        df_summary = pd.DataFrame(data={'track_7digitalid': dataset_1['track_7digitalid'], 'track_id': dataset_2['track_id']})
        df_summary['track_id'] = df_summary['track_id'].apply(lambda x: x.decode('UTF-8'))
        return df_summary

def find_tracks():
    paths = []
    for folder, subfolders, files in os.walk(root_dir):
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
        'size': float
            The file size of the mp3 file.
        
    
    '''
    s = []
    paths = df['path'] #more efficient
    for path in paths: 
        #path = os.path.join(root_dir, path)
        path = root_dir[:-1]+ path #was wrong, now fixed
        s.append(os.path.getsize(path))
    #df['size'] = pd.Series(s, index=df.index) # sizes is better since df.size is ambiguous...
    df['sizes'] = pd.Series(s, index=df.index)
    return df

def check_mutagen_info(df, add_length=True, add_channels=True, verbose=True,
                       save_csv=True, output_path='/srv/data/urop/ultimate_csv_size.csv'): 
    # You will want to see the progress, and after this long conversion, you may want to save it.
    '''
    Parameters
    ----------
    df: pd.DataFrame
        The input dataframe which you want extra information (length and number 
        of channel of tracks).
    
    add_length: bool
        If true, the computed lengths column is appended to the df.
    
    add_channels: bool
        If true, the computed number of channels column is appended to the df.
        
    verbose: bool
        If true, progress of running the program is printed.
        
    save_csv: bool
        If true, the resulting dataframe is saved as a csv file.
        
    output_dir: str
        The output path of the csv saved if save_csv is True.
        
        
    Returns
    -------
    df: pd.DataFrame
        A dataframe that has two extra columns if add_length and add_channels
        are set to be True:
        'lengths': float
            The duration of the mp3 tracks.
            
        'channels': float
            The number of channels of the mp3 tracks.
            
        NOTE: empty cell is returned to the corresponding rows for lengths 
        and channels if the script cannot read the size of the tracks or cannot 
        open the tracks (i.e. broken tracks).
        
    csv: 
        The df is saved as csv if save_csv is True.
        
    
    '''
    
    tot = len(df)
    #mod = len(df) // 100    #len(df) is not divisiable by 100..
    l = []
    c = []
    for idx, path in enumerate(df['path']):
        #path = os.path.join(root_dir, path)
        path = root_dir[:-1]+ path #was wrong, now fixed
        try:
            audio = mutagen.mp3.MP3(path)
            l.append(audio.info.length)  #
            c.append(audio.info.channels) #
        except:
            l.append('')
            c.append('')
            continue
        #l.append(audio.info.length)
        #c.append(audio.info.channels) This is wrong I think
        
        if verbose == True:
            if idx % 1000 == 0: # change based on comment above
                print('PROGRESS: {:6d}/{:6d}'.format(idx, tot))

    if add_length == True: 
        #df['length'] = pd.Series(l, index=df.index)
        df['lengths'] = pd.Series(l, index=df.index) # sizes is better since df.length is ambiguous...
    if add_channels == True:
        df['channels'] = pd.Series(c, index=df.index)
        
        if save_csv:   #added
            df.to_csv(output_path, index=False)
    return df

def die_with_usage():
    print()
    print("track_fetch.py - Script to search for MP3 files within root_dir and output a CSV file with (optionally) the")
    print("                 following columns: track 7digitalID, path, file size, track length, number of channels.")
    print()
    print("Usage:     python track_fetch.py <output filename> [options]")
    print()
    print("General Options:")
    print("  --no-size              Do not add column containing file sizes to output file.")
    print("  --no-length            Do not add column containing track lengths to output file.")
    print("  --no-channels          Do not add column containing track number of channels to output file.")
    print("  --root-dir             Set different root_dir.")
    print("  --help                 Show this help message and exit.")
    print("  --verbose              Show progress.")
    print()
    print("Example:   python track_fetch.py ./tracks_on_boden.csv --root-dir /data/songs/ --no-channels --verbose")
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
        if add_size == True:
            df = check_size(df)
        if add_length == True or add_channels == True:
            df = check_mutagen_info(df, add_length, add_channels, verbose)
        df.to_csv(output, index=False)