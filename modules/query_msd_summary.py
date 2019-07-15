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

path_h5 = '/srv/data/msd/msd_summary_file.h5' # default path to msd summary file

path_db = '/srv/data/urop/track_metadata.db'  # default path to 'database version' of msd summary file

def set_path_h5(new_path):
    global path_h5
    path_h5 = new_path

def set_path_db(new_path):
    global path_db
    path_db = new_path

def get_trackid_from_7digitalid(*ids):
    ''' Returns the track_id of the song specified by the 7digital_id. '''

    if len(ids) == 1 and hasattr(ids[0], '__iter__') and not isinstance(ids[0], str):
        ids = ids[0]

    with tables.open_file(path_h5, mode='r') as f:
        output = []

        if all([isinstance(id, int) for id in ids]):
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

def get_7digitalid_from_trackid(*ids):
    ''' Returns the 7digital_id of the song specified by the track_id. '''

    if len(ids) == 1 and hasattr(ids[0], '__iter__') and not isinstance(ids[0], str):
        ids = ids[0]

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

def get_attribute(attr: str, *ids):
	''' Returns a list with the desired attribute given either the track_id or the song_id (or really anything else with which you can windex our SQL database...).
    
    - 'id': is the track_id or the song_id we are using to query the database;
    - 'id_type': is the type of id we are using;
    - 'desired_column': is the song attribute we are looking for (that is, one of the database columns).
    
    EXAMPLE: get_attribute('SOBNYVR12A8C13558C', 'song_id') --> [('Si Vos Querés',)].
    '''

	if len(ids) == 1 and hasattr(ids[0], '__iter__') and not isinstance(ids[0], str):
		ids = ids[0]

	id_type = ('track_id', 'song_id')

	if all([isinstance(id, int) for id in ids]):
		ids = [get_trackid_from_7digitalid(id) for id in ids]
		id_type = id_type[0]
	elif all([id[:2] == 'TR' for id in ids]):
		id_type = id_type[0]
	elif all([id[:2] == 'SO' for id in ids]):
		id_type = id_type[1]
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