import h5py
import mutagen.mp3
import os
import pandas as pd
import sys

root_dir = '/srv/data/msd/7digital/'

def set_mp3_root_dir(new_root_dir):
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
    s = []
    for path in df['path']:
        path = os.path.join(root_dir, path)
        s.append(os.path.getsize(path))
    df['size'] = pd.Series(s, index=df.index)
    return df

def check_mutagen_info(df, add_length=True, add_channels=True, verbose=False):
    tot = len(df)
    mod = len(df) // 100
    l = []
    c = []
    for idx, path in enumerate(df['path']):
        path = os.path.join(root_dir, path)
        try:
            audio = mutagen.mp3.MP3(path)
        except:
            l.append('')
            c.append('')
            continue
        l.append(audio.info.length)
        c.append(audio.info.channels)
        
        if verbose == True:
            if idx % mod == 0:
                print('PROGRESS: {:6d}/{:6d}'.format(idx, tot))

    if add_length == True: 
        df['length'] = pd.Series(l, index=df.index)
    if add_channels == True:
        df['channels'] = pd.Series(c, index=df.index)
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