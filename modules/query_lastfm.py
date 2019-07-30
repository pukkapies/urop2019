''' Contains simple tools for querying the lastfm_tags.db file


Notes
-----
The database contains 3 tables: tids, tags, tid_tag.
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

default = '/srv/data/msd/lastfm/SQLITE/lastfm_tags.db'

class LastFm:
    ''' Opens a SQLite connection to the last.fm database. Provides methods to perform advanced queries on it.

    Methods
    -------
    - tid_to_tid_num
        Get tid_num given tid.

    - tid_num_to_tid
        Get tid given tid_num.

    - tid_num_to_tag_num
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

    def __init__(self, path = default):
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

    def tid_num_to_tag_num(self, tid_num):
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


class LastFm2Pandas():
    ''' Reads the last.fm database into pandas dataframes. Provides methods to perform advanced queries on it.

    Methods
    -------
    - tid_to_tid_num
        Return tid_num(s) given tid(s).

    - tid_num_to_tid
        Return tid(s) given tid_num(s).

    - tid_num_to_tag_nums
        Return tag_num(s) given tid_num(s).

    - tid_num_to_tag
        Return tag(s) given tid_num(s).

    - tag_num_to_tag
        Return tag(s) given tag_num(s).

    - tag_to_tag_num
        Return tag_num(s) given tag(s).

    - tid_num_to_tags
        Get tags for given tid_num(s).

    - tid_to_tags
        Get tags for given tid(s).
    
    - genre
        Get all tids which have some certain tag(s).

    - genre_not_genre
        Get all tids which have some certain tag(s) but not some other(s).
    '''

    def __init__(self, path = default, no_tags = False, no_tids = False, no_tid_tag = False):
        '''
        Parameters
        ----------
        path : str
            Path to tags database. Defaults to path on Boden.

        no_tags : bool
            If True, do not store tags table.

        no_tids : bool
            If True, do not store tids table.

        no_tid_tag : bool
            If True, do not store tid_tag table.
        '''

        conn = sqlite3.connect(path)

        # open tables as dataframes and shift index to match rowid in database
        if not no_tags:
            self.tags = pd.read_sql_query('SELECT *  FROM tags', conn)
            self.tags.index += 1
        if not no_tids:
            self.tids = pd.read_sql_query('SELECT * FROM tids', conn)
            self.tids.index += 1
        if not no_tid_tag:
            self.tid_tag = pd.read_sql_query('SELECT * FROM tid_tag', conn)
            self.tid_tag.index += 1

        conn.close()

    def tid_to_tid_num(self, tid):
        ''' Returns tid_num(s) given tid(s)
        
        Parameters
        ----------
        tid : str, array-like
            A single tid or an array-like structure containing tids

        Returns
        -------
        tid_num : str, pandas.Index
            if tid is a string: 
                corresponding tid_num (int)
            if array_like: 
                pd.Index object containing tid_nums
        '''

        if isinstance(tid, str):
            return self.tids.loc[self.tids.tid == tid].index[0]

        return self.tids.loc[self.tids.tid.isin(tid)].index.tolist()

    def tid_num_to_tid(self, tid_num):
        ''' Returns tid(s) given tid_num(s)
        
        Parameters
        ----------
        tid_num : int, array-like
            A single tid_num or an array-like structure containing tid_nums.

        Returns
        -------
        tid : str, ndarray
            if tid_num is an int: 
                corresponding tid (str)
            if array-like: 
                ndarray containing corresponding tids
        '''

        if isinstance(tid_num, (int, np.integer)):
            return self.tids.at[tid_num, 'tid']

        return self.tids.loc[self.tids.index.isin(tid_num), 'tid'].values

    def tid_num_to_tag_nums(self, tid_num):
        ''' Returns tag_nums given tid_num(s)
        
        Parameters
        ----------
        tid_num : int, array-like
            A single tid_num or an array-like structure containing tid_nums

        Returns
        -------
        tag_nums : ndarray, series.
            if tid_num is an int:
                ndarray of corresponding tag_nums (int)
            if array-like:
                pd.Series where indices are tid_nums and values are
                corresponding list of tag_nums
        '''
        
        if isinstance(tid_num, (int, np.integer)):
            return self.tid_tag.loc[self.tid_tag.tid == tid_num, 'tag'].values

        tag_nums = [self.tid_tag.loc[self.tid_tag.tid == num, 'tag'].values for num in tid_num]
        return pd.Series(tag_nums, index=tid_num)

    def tag_num_to_tag(self, tag_num):
        ''' Returns tag(s) given tag_num(s) 

        Parameters
        ----------
        tag_num : int, array-like
            A single tag_num or an array-like structure containing tag_nums

        Returns
        -------
        tag : str, ndarray
            if tid_num is an int:
                corresponding tag (str)
            if array-like:
                ndarray containing corresponding tags
        '''

        if isinstance(tag_num, (int, np.integer)):
            return self.tags.at[tag_num, 'tag']

        return self.tags.loc[self.tags.index.isin(tag_num), 'tag'].values

    def tag_to_tag_num(self, tag):
        ''' Returns tag_num(s) given tag(s)

        Parameters
        ----------
        tag : str, array-like
            A single tag or an array-like structure containing tags

        Returns
        -------
        tag_num : int, pd.Index
            if tag is a str:
                Corresponding tag_num (int)
            if array-like:
                pd.Index containing corresponding tag_nums
        '''

        if isinstance(tag, str):
            return self.tags.loc[self.tags.tag == tag].index[0]

        return self.tags.loc[self.tags.tag.isin(tag)].index

    def tid_num_to_tags(self, tid_num):
        ''' Gets tags for given tid_num(s) 
        
        Parameters
        ----------
        tid_num : int, array-like
            A single tid_num or an array-like structure containing tid_nums

        Returns
        -------
        tags : ndarray, pd.Series 
            if tid_num is an int:
                ndarray containing corresponding tags     
            if array-like:
                pd.Series having tid_nums as indices and list of tags as values
        '''

        tag_nums = self.tid_num_to_tag_nums(tid_num)

        if isinstance(tag_nums, (list, np.ndarray)):
            return self.tag_num_to_tag(tag_nums)

        return tag_nums.map(self.tag_num_to_tag)

    def tid_to_tags(self, tid):
        ''' Gets tags for given tid(s) 
        
        Parameters
        ----------
        tid : str, array-like
            A single tid or an array-like structure containing tids

        Returns
        -------
        tags : ndarray, pd.Series
            if tag is a str:
                ndarray containing corresponding tags
            if array-like:
                pd.Series having tids as indices and list of tags as values
        '''

        tid_num = self.tid_to_tid_num(tid)
        tags = self.tid_num_to_tags(tid_num)

        if isinstance(tags, (list, np.ndarray)):
            return tags

        return tags.rename(self.tid_num_to_tid)