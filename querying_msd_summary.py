import tables

path = '/srv/msd/msd_summary_file.h5'

def title_from_7digitalid(id: int):
    with tables.open_file(path, mode='r') as f:
        idxs = f.root.metadata.songs.get_where_list('track_7digitalid==' + str(id))
        return f.root.metadata.songs[idxs]['title']