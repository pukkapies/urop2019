''' Contains tools to query the msd_summary_file.h5 file.


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

IMPORTANT: If using this script elsewhere than on Boden then run set_path_h5(path) or
set_path_db(path) to set the path of the summary files. Otherwise it will use the default 
path, which is the path to the database on Boden.


Functions
---------
- set_path_h5
    Set path to the msd_summary_file.h5.

- set_path_db
    Set path to the track_metadata.db.

- get_trackid_from_7digitalid
    Return a (list of) tid('s) given a (list of) 7digitalid('s).

- get_7digitalid_from_trackid
    Return a (list of) 7digitalid('s) given a (list of) tid('s).

- get_attribute
    Return a (list of) attribute(s) such as title or artist_id given a (list of ) track_id('s), song_id('s) or 7digitalid('s)
'''

import os
import sqlite3

import h5py
import numpy as np

path_h5 = '/srv/data/msd/msd_summary_file.h5' # default path to msd summary file
path_db = ''                                  # default path to msd summary file (in its .db version)

def set_path_h5(new_path):
    global path_h5
    path_h5 = os.path.expanduser(new_path)

def set_path_db(new_path):
    global path_db
    path_db = os.path.expanduser(new_path)

def get_trackid_from_7digitalid(ids: list):
    ''' Returns the track_id of the song specified by the 7digital_id. '''

    if isinstance(ids, int) or not hasattr(ids, '__iter__'): ids = [ids]

    with h5py.File(path_h5, 'r') as f:

        dataset_1 = f['metadata']['songs']
        dataset_2 = f['analysis']['songs']

        output = {}

        for id in ids:
            tid = dataset_2[np.where(dataset_1['track_7digitalid'] == id)[0][0]]['track_id'].decode('UTF-8')
            output[id] = tid

        if len(output) > 1:
            return output
        else:
            return output[id]

def get_7digitalid_from_trackid(ids: list):
    ''' Returns the 7digital_id of the song specified by the track_id. '''

    if isinstance(ids, str) or not hasattr(ids, '__iter__'): ids = [ids]

    with h5py.File(path_h5, 'r') as f:

        dataset_1 = f['metadata']['songs']
        dataset_2 = f['analysis']['songs']

        output = {}

        for id in ids:
            tid = dataset_1[np.where(dataset_2['track_id'] == id.encode('UTF-8'))[0][0]]['track_7digitalid']
            output[id] = tid

        if len(output) > 1:
            return output
        else:
            return output[id]

def get_attribute(attr: str, ids: list):
    ''' Returns the specified attribute corresponding to each track passed as input.
    
    Parameters
    ----------
    attr: str
        The column of the track_metadata.db database to be queried. 
        Possible values are 'title', 'release', 'duration', 'artist_id', 'artist_mbid', 
        'artist_name', 'artist_familiarity', 'artist_hotttnesss', 'year'.

    Returns
    -------
    list or str
        List of attributes for each track passed as input. Sigle attribute
        if only one track is passed.

    Examples
    --------
    >>> get_attribute('title', 'TRMMQYP128F428E72A')
    'Shine On'

    >>> get_attribute('artist_name', 'SOGTUKN12AB017F4F1')
    'Hudson Mohawke'

    >>> tids = get_trackid_from_7digitalid([2168257, 2264873])
    >>> get_attribute('title', tids)
    [Si Vos Querés, 'Tangle of Aspens']

    >>> get_attribute('title', [2168257, '2264873'])
    [Si Vos Querés, 'Tangle of Aspens']
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
