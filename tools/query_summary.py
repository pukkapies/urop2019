'''
Davide Gallo (2019) Imperial College London
dg5018@ic.ac.uk


This code contains a set of getters functions to retrieve
various song attributes from the HDF5 summary file, given the 
song 7digital ID.

The track_metadata.db database contains a table called 'songs'
with the following columns: (track_id, title, song_id, 
release, artist_id, artist_mbid, artist_name, 
duration, artist_familiarity, artist_hotttnesss, year). Using the
database to retrieve song attributes is much faster than
scanning the HDF5 summary file.

The database can be downloaded from: http://www.ee.columbia.edu/~thierry/track_metadata.db

Since the database does not contain a '7digital_id' column, we
have to go through the whole HDF5 file to get match a 7digital_id
with its track_id.


Copyright 2019, Davide Gallo <dg5018@ic.ac.uk>
'''

import tables, sqlite3

PATH_TO_H5 = '/srv/data/msd/msd_summary_file.h5'

def get_trackid_from_7digitalid(id: int, filepath: str = PATH_TO_H5):
    ''' Returns the track_id of the song specified by the 7digital_id.
    '''
    with tables.open_file(filepath, mode='r') as f:
        idx = f.root.metadata.songs.get_where_list('track_7digitalid==' + str(id))

        # check whether the 7digital_id corresponds to one and only one track
        assert len(idx) == 1

        return f.root.analysis.songs[idx]['track_id'][0].decode('UTF-8')

def get_7digitalid_from_trackid(id: str, filepath: str = PATH_TO_H5):
    ''' Returns the 7digital_id of the song specified by the track_id.
    '''
    with tables.open_file(filepath, mode='r') as f:
        idx = f.root.analysis.songs.get_where_list('track_id=="' + id + '"')

        # check whether the 7digital_id corresponds to one and only one track
        assert len(idx) == 1

        return f.root.metadata.songs[idx]['track_7digitalid'][0]

PATH_TO_DB = '/srv/data/urop/track_metadata.db'

def get_attribute(id: str, id_type: str = 'track_id', desired_column: str = 'title', filepath: 'str' = PATH_TO_DB):
    ''' Returns a list with the desired attribute given either the track_id or the song_id (or really anything else with which you can windex our SQL database...).
    
    - 'id': is the track_id or the song_id we are using to query the database;
    - 'id_type': is the type of id we are using;
    - 'desired_column': is the song attribute we are looking for (that is, one of the database columns).
    
    EXAMPLE: get_attribute('SOBNYVR12A8C13558C', 'song_id') --> [('Si Vos Querés',)].
    '''
    conn = sqlite3.connect(filepath)
    q = "SELECT " + desired_column + " FROM songs WHERE " + id_type + " = '" + id  + "'"
    res = conn.execute(q)
    return res.fetchall()

import h5py, pandas as pd

def extract_7did_tid_from_h5(filepath: str = PATH_TO_H5):
	pass