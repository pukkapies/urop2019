''' Contains simple tools for querying the lastfm_tags.db file


Notes
-----
The lastfm database contains 3 tables: tids, tags, tid_tag.
- tids, 1-column table containing the track ids.
- tid_tags, contains 3 columns:
    - tid: rowid of the track id in the tids table.
    - tag: rowid of the tag that belongs to the tid in the same row.
    - val: number between 0 and 100 (guessing this is how accurate the tag is?)
- tags, 1-column table containing the tags.

IMPORTANT: If using this script elsewhere than on Boden then run set_path(new_path) to
set the path of the database. Otherwise it will use the default path, which is the path
to the database on Boden.


Classes
-------
- LastFm
    Open a connection to the db file and provide methods to perform queries on it.
    This class is faster to init, but some queries (expecially on the tid_tag table) might take some time to perform.

- LastFm2Pandas
    Read the database into three pandas dataframes and prodive methods to retrieve information from them.
    This class is slower to init, since the whole database is loaded into memory, but consequently queries are much faster. This class also contain some additional "advanced" methods.
'''

import sqlite3

class LastFm:
    ''' Opens a SQLite connection to the last.fm database. Provides methods to perform advanced queries on it.

    Methods
    -------
    - tid_to_tid_nums
        Get tid_num given tid.

    - tid_num_to_tid
        Get tid given tid_num.

    - tid_num_to_tag_nums
        Get tag_num given tid_num.

    - tag_num_to_tag
        Get tag given tag_num.

    - tag_to_tag_num
        Get tag_num given tag.

    - query_tags
        Get a list of tags associated to given tid.

    - query_tags_dict
        Get a dict with tids as keys and a list of its tags as value.

    - tid_tag_count
        Get a dict with tids as keys and its number of tags as value.

    - filter_tags
        Filter list of tids based on minimum number of tags.

    - tag_count
        Get a dict with the tags associated to tids as keys and their count number as values.
    '''

    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.c = self.conn.cursor()
    
    def __del__(self): # close the connection gracefully when the object goes out of scope
        self.conn.close()

    def query(self, query, *parameters):
        self.c.execute(query, parameters)

    def tid_to_tid_num(self, tid):
        ''' Returns tid_num, given tid. '''

        q = "SELECT rowid FROM tids WHERE tid = ?"
        self.query(q, tid)
        return self.c.fetchone()[0]

    def tid_num_to_tid(self, tid_num):
        ''' Returns tid, given tid_num. '''

        q = "SELECT tid FROM tids WHERE rowid = ?"
        self.query(q, tid_num)
        return self.c.fetchone()[0]

    def tid_num_to_tag_nums(self, tid_num):
        ''' Returns list of the associated tag_nums to the given tid_num. '''

        q = "SELECT tag FROM tid_tag WHERE tid = ?"
        self.query(q, tid_num)
        return [i[0] for i in self.c.fetchall()]
        
    def tag_num_to_tag(self, tag_num):
        ''' Returns tag given tag_num. '''

        q = "SELECT tag FROM tags WHERE rowid = ?"
        self.query(q, tag_num)
        return self.c.fetchone()[0]

    def tag_to_tag_num(self, tag):
        ''' Returns tag_num given tag. '''

        q = "SELECT rowid FROM tags WHERE tag = ?"
        self.query(q, tag)
        return self.c.fetchone()[0]

    def get_tags(self):
        ''' Returns a list of all the tags. '''

        q = "SELECT tag FROM tags"
        self.query(q)
        return [i[0] for i in self.c.fetchall()]

    def get_tag_nums(self):
        ''' Returns a list of all the tag_nums. '''

        q = "SELECT rowid FROM tags"
        self.query(q)
        return [i[0] for i in self.c.fetchall()]

    def get_tids(self):
        ''' Gets tids which have at least one tag. '''

        q = "SELECT tid FROM tids WHERE tid IS NOT NULL"
        self.query(q)
        return [i[0] for i in self.c.fetchall()]

    def get_tid_nums(self):
        ''' Gets tid_num of tids which have at least one tag. '''

        q = "SELECT rowid FROM tids WHERE tid IS NOT NULL"
        self.query(q)
        return [i[0] for i in self.c.fetchall()]

    def fetch_all_tids_tags(self):
        ''' Returns a list of tuples containing tids and tags (as they appear in the tid_tag table). '''

        q = "SELECT tid, tag FROM tid_tag"
        self.query(q)
        return self.c.fetchall()

    def fetch_all_tids_tags_threshold(self, threshold = 0):
        ''' Returns a list of tuples containing tids and tags (as they appear in the tid_tag table) satisfying val > threshold. '''

        q = "SELECT tid, tag FROM tid_tag WHERE val > ?"
        self.query(q, threshold)
        return self.c.fetchall()

    def query_tags(self, tid):
        ''' Gets tags for a given tid. '''
        
        tags = []
        for tag_num in self.tid_num_to_tag_nums(self.tid_to_tid_num(tid)):
            tags.append(self.tag_num_to_tag(tag_num))
        return tags

    def query_tags_dict(self, tids): # pandas series would perform a bit faster here...
        ''' Gets tags for a given list of tids.
        
        Parameters
        ----------
        tids : list
            List containing tids as strings.

        Returns
        -------
        tag_dict : dict
            The keys are the tids from the input list.
            The values are lists of tags for each given tid.
        '''

        tags_dict = {}
        for tid in tids:
            tags_dict[tid] = self.query_tags(tid)
        return tags_dict

    def tag_count(self, tids):
        ''' Gets number of tags for each given tid.
        
        Parameters
        ----------
        tids : list
            List containing tids as strings.

        Returns
        -------
        count_dict : dict
            The keys are the tags associated to any tid from the input list.
            The values are the number of tids which the given tag is associated to.
        '''

        count_dict = {}
        for tag_list in self.query_tags_dict(tids).values():
            for tag in tag_list:
                if tag not in count_dict:
                    count_dict[tag] = 1
                else:
                    count_dict[tag] += 1 
        return count_dict

    def tid_tag_count(self, tids):
        ''' Gets number of tags for each given tid.
        
        Parameters
        ----------
        tids : list
            List containing tids as strings.

        Returns
        -------
        count_dict : dict
            The keys are the tids from the input list.
            The values are the number of tags for each given tid.
        '''

        count_dict = {}
        for tid in tids:
            count_dict[tid] = len(self.query_tags(tid))
        return count_dict

    def filter_tags(self, tids, min_tags):
        ''' Given list of tids, returns list of those with more than min_tags tags. '''

        count_dict = self.tid_tag_count(tids)
        tids_filtered = [tid for tid in tids if count_dict[tid] >= min_tags]
        return tids_filtered