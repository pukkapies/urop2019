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

path_to_h5 = '/srv/data/msd/msd_summary_file.h5'

def from_7digitalid_get_trackid(id: int):
    ''' Returns a numpy.ndarray with the track_id of the song specified by the 7digital_id.
    '''
    with tables.open_file(path_to_h5, mode='r') as f:
        idxs = f.root.metadata.songs.get_where_list('track_7digitalid==' + str(id))
        return f.root.analysis.songs[idxs]['track_id']

path_to_db = '/srv/data/urop/track_metadata.db'

def get_attribute(id: str, id_type: str = 'track_id', desired_column: str = 'title'):
    ''' Returns a list with the desired attribute given either the track_id or the song_id (or really anything else with which you can windex our SQL database...).
    
    - 'id': is the track_id or the song_id we are using to query the database;
    - 'id_type': is the type of id we are using;
    - 'desired_column': is the song attribute we are looking for (that is, one of the database columns).
    
    EXAMPLE: get_attribute('SOBNYVR12A8C13558C', 'song_id') --> [('Si Vos Querés',)].
    '''
    conn = sqlite3.connect(path_to_db)
    q = "SELECT " + desired_column + " FROM songs WHERE " + id_type + " ='" + id  + "'"
    res = conn.execute(q)
    return res.fetchall()