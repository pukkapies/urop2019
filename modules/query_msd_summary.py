''' Contains simple tools for querying the msd_summary_file.h5 file


Notes
-----
The track_metadata database contains a table called 'songs' with the following 
columns: track_id, title, song_id, release, artist_id, artist_mbid, artist_name, duration, 
artist_familiarity, artist_hotttnesss, year. These are the possible values of
the variable attr in the function get_attribute(). Querying the database is much
faster than searching through the h5 file (which is not indexed).

To get the track_id associated with a given 7digital_id, the only way is to search
through the h5 file (it might take some seconds), since we don't have a '7digital_id'
column on the database. Yes, we could have added it. Still, this module is mainly
for experimentation, and it will not be used in production.

The function get_attribute() automatically infers whether a track_id or a song_id
is being used. If a 7digital_id is begin used, get_trackid_from_7digitalid() is
called behind the scenes (therefore making the execution much slower). The function is
extremely flexible, and can receive multiple id's as input, both as a single tuple/list 
or as multiple arguments.

IMPORTANT: If using this script elsewhere than on boden then run set_path_h5(path) or
set_path_db(path) to set the path of the summary files. Otherwise it will use the default 
path, which is the path to the database on boden.


Functions
---------
- set_path_h5                   Sets path to the msd_summary_file.h5
- set_path_db                   Sets path to the track_metadata.db
- get_trackid_from_7digitalid   Returns a (list of) tid(s) given a (list of) 7digital_id(s)
- get_7digitalid_from_trackid   Returns a (list of) 7digital_id(s) given a (list of) tid(s)
- get_attribute                 Returns a (list of) attribute(s) such as title or artist_id given a (list of )track_id(s), song_id(s) or 7digital_id(s)
'''

import tables, sqlite3

path_h5 = '/srv/data/msd/msd_summary_file.h5' # default path to msd summary file

path_db = '/srv/data/urop/track_metadata.db'  # default path to 'database version' of msd summary file

def set_path_h5(new_path):
    global path_h5
    path_h5 = new_path

def set_path_db(new_path):
    global path_db
    path_db = new_path

def get_trackid_from_7digitalid(ids: list):
    ''' Returns the track_id of the song specified by the 7digital_id. '''

    if isinstance(ids, str) or not hasattr(ids, '__iter__'): ids = [ids]

    with tables.open_file(path_h5, mode='r') as f:
        output = []
        ids = [str(id) for id in ids]
        for id in ids:
            idx = f.root.metadata.songs.get_where_list('track_7digitalid==' + id)

            # check whether the given id corresponds to one and only one track
            assert len(idx) == 1

            tid = f.root.analysis.songs[idx]['track_id'][0].decode('UTF-8')
            output.append(tid)
        
        if len(output) > 1:
            return output
        else:
            return output[0]

def get_7digitalid_from_trackid(ids: list):
    ''' Returns the 7digital_id of the song specified by the track_id. '''

    if isinstance(ids, str) or not hasattr(ids, '__iter__'): ids = [ids]

    with tables.open_file(path_h5, mode='r') as f:
        output = []
        for id in ids:
            idx = f.root.analysis.songs.get_where_list('track_id=="' + id + '"')

            # check whether the given id corresponds to one and only one track
            assert len(idx) == 1

            tid = f.root.metadata.songs[idx]['track_7digitalid'][0]
            output.append(tid)
        
        if len(output) > 1:
            return output
        else:
            return output[0]

def get_attribute(attr: str, ids: list):
    ''' Returns the specified attribute corresponding to each track passed as input.
    
    Parameters
    ----------
    attr : str
        the column of the track_metadata.db database to be queried (possible
        values are 'title', 'release', 'duration', 'artist_id', 'artist_mbid',
        'artist_name', 'artist_familiarity', 'artist_hotttnesss', 'year')
    '''

    if isinstance(ids, str) or not hasattr(ids, '__iter__'): ids = [ids]

    if all([isinstance(id, int) for id in ids]):
        ids = [get_trackid_from_7digitalid(id) for id in ids]
        id_type = 'track_id'
    elif all([id[:2] == 'TR' for id in ids]):
        id_type = 'track_id'
    elif all([id[:2] == 'SO' for id in ids]):
        id_type = 'song_id'
    else:
        raise NameError

    output = []
    conn = sqlite3.connect(path_db)
    c = conn.cursor()
    q = "SELECT " + attr + " FROM songs WHERE " + id_type + " = '"

    for id in ids:
        c.execute(q + id + "'")
        output.extend(c.fetchone())

    conn.close()

    if len(output) > 1:
        return output
    else:
        return output[0]
