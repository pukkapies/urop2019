''' Contains tools to query the lastfm_tags.db database.


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
    Read the databsdr and provide methods to perform queries on it.
    This class is faster to init, but some queries (expecially on the tid_tag table) might take some time to perform.

- LastFm2Pandas
    Read the database into three pandas dataframes and provide methods to perform queries on it.
    This class is slower to init, since the whole database is loaded into memory, but consequently queries are much faster. This class also contain some additional "advanced" methods.

- Matrix
    Read the database from either LastFm or LastFm2Pandas, and perform numerical analyses of how tags are distributed among tracks.

Functions
---------
- crazysum
    Compute the sum of the first n s-gonal numbers in n-dimensions.
'''

import itertools
import math
import os
import pickle
import sqlite3

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import sparse

from tensorflow.keras.utils import Progbar

DEFAULT = '/srv/data/msd/lastfm/SQLITE/lastfm_tags.db'

class LastFm():
    ''' Reads the last.fm database and provides methods to perform advanced queries on it.

    Methods
    -------
    - to_csv
        Convert the tags database into three different .csv files.

    - sql_tag_num_to_tag
        Get tag given tag_num.
    
    - tag_num_to_tag
        Get tag given tag_num (vectorized version).

    - sql_tag_to_tag_num
        Get tag_num given tag.
    
    - tag_to_tag_num
        Get tag_num given tag (vectorized version).

    - sql_tid_num_to_tid
        Get tid given tid_num.

    - tid_num_to_tid
        Get tid given tid_num (vectorized version).

    - sql_tid_to_tid_num
        Get tid_num given tid.
    
    - tid_to_tid_num
        Get tid_num given tid (vectorized version).

    - tid_num_to_tag_nums
        Get tag_num given tid_num.

    - get_tags
        Return a list of all the tags.
        
    - get_tag_nums
        Return a list of all the tag_nums.

    - get_tids
        Get tids which have at least one tag.
        
    - get_tid_nums
        Get tid_num of tids which have at least one tag.

    - query_tags
        Get tags for given tid.

    - query_tags_dict
        Get a dict with tids as keys and a list of its tags as value.

    - tid_tag_count
        Given a list of tids, returns a dict with tids as keys and its number of tags as value.

    - tid_tag_count_filter
        Given a list of tids, filters out those with less than minimum number of tags.

    - fetch_all_tids_tags
        Return a dataframe containing tids and tags (as they appear in the tid_tag table).
        
    - fetch_all_tids_tags_threshold
        Return a dataframe containing tids and tags (as they appear in the tid_tag table) satisfying val > threshold.
    
    - with_tag
        Return all tids with a given tag.

    - popularity
        Return a dataframe containing the tags ordered by popularity, together with the number of times they appear.
    '''

    def __init__(self, path):
        if not os.path.isfile(path):
            raise OSError("file " + path + " does not exist!")

        self.conn = sqlite3.connect(path)
        self.c = self.conn.cursor()
        self.path = path

        self.tag_num_to_tag_vec = np.vectorize(self.sql_tag_num_to_tag)
        self.tag_to_tag_num_vec = np.vectorize(self.sql_tag_to_tag_num)
        self.tid_num_to_tid_vec = np.vectorize(self.sql_tid_num_to_tid)
        self.tid_to_tid_num_vec = np.vectorize(self.sql_tid_to_tid_num)

    def __del__(self): # close the connection gracefully when the object goes out of scope
        self.conn.close()

    def _query(self, q, *params):
        self.c.execute(q, params)
    
    def to_csv(self, output_dir=None):
        ''' Converts the tags database into three different .csv files. 
        
        Parameters
        ----------
        output_dir: str
            Output directory of the .csv files. If None, the files will be saved
            under the same directory as lastfm_tags.db
        '''
        
        if output_dir is None:
            output_dir = os.path.dirname(self.path)

        q = "SELECT name FROM sqlite_master WHERE type='table'"
        self._query(q)
        tables = [i[0] for i in self.c.fetchall()]
        for table in tables:
            print('saving '+ 'lastfm' + '_' + table +'.csv')
            path = os.path.join(output_dir, 'lastfm' + '_' + table +'.csv')
            df = pd.read_sql_query("SELECT * FROM " + table, self.conn)
            df.to_csv(path, index_label=False)
        return
        
    def sql_tag_num_to_tag(self, tag_num):
        ''' Returns tag given tag_num. '''

        q = "SELECT tag FROM tags WHERE rowid = ?"
        self._query(q, tag_num)
        return self.c.fetchone()[0]
    
    def tag_num_to_tag(self, tag_num):
        ''' Returns tag given tag_num. '''

        if np.issubdtype(type(tag_num), np.integer):
            return self.sql_tag_num_to_tag(str(tag_num))
        else:
            return self.tag_num_to_tag_vec([str(i) for i in tag_num])

    def sql_tag_to_tag_num(self, tag):
        ''' Returns tag_num given tag. '''
        
        q = "SELECT rowid FROM tags WHERE tag = ?"
        self._query(q, tag)
        return self.c.fetchone()[0]
    
    def tag_to_tag_num(self, tag):
        ''' Returns tag_num given tag. '''

        if isinstance(tag, str):
                return self.sql_tag_to_tag_num(tag)
        else:
            return self.tag_to_tag_num_vec(tag)

    def sql_tid_num_to_tid(self, tid_num):
        ''' Returns tid, given tid_num. '''

        q = "SELECT tid FROM tids WHERE rowid = ?"
        self._query(q, tid_num)
        return self.c.fetchone()[0]
    
    def tid_num_to_tid(self, tid_num):
        ''' Returns tid, given tid_num. '''

        if np.issubdtype(type(tid_num), np.integer):
            return self.sql_tid_num_to_tid(str(tid_num))
        else:
            return self.tid_num_to_tid_vec([str(i) for i in tid_num])

    def sql_tid_to_tid_num(self, tid):
        ''' Returns tid_num, given tid. '''

        q = "SELECT rowid FROM tids WHERE tid = ?"
        self._query(q, tid)
        return self.c.fetchone()[0]
    
    def tid_to_tid_num(self, tid):
        ''' Returns tid_num, given tid. '''

        if isinstance(tid, str):
            return self.sql_tid_to_tid_num(tid)
        else:
            return self.tid_to_tid_num_vec(tid)

    def tid_num_to_tag_nums(self, tid_num):
        ''' Returns list of the associated tag_nums to the given tid_num. '''

        q = "SELECT tag FROM tid_tag WHERE tid = ?"
        self._query(q, tid_num)
        return [i[0] for i in self.c.fetchall()]

    def get_tags(self):
        ''' Returns a list of all the tags. '''

        q = "SELECT tag FROM tags"
        self._query(q)
        return [i[0] for i in self.c.fetchall()]

    def get_tag_nums(self):
        ''' Returns a list of all the tag_nums. '''

        q = "SELECT rowid FROM tags"
        self._query(q)
        return [i[0] for i in self.c.fetchall()]

    def get_tids(self):
        ''' Gets tids which have at least one tag. '''

        q = "SELECT tid FROM tids WHERE tid IS NOT NULL"
        self._query(q)
        return [i[0] for i in self.c.fetchall()]

    def get_tid_nums(self):
        ''' Gets tid_num of tids which have at least one tag. '''

        q = "SELECT rowid FROM tids WHERE tid IS NOT NULL"
        self._query(q)
        return [i[0] for i in self.c.fetchall()]

    def fetch_all_tids_tags(self):
        ''' Returns a list of tuples containing tids and tags (as they appear in the tid_tag table). '''

        q = "SELECT tid, tag FROM tid_tag"
        self._query(q)
        return pd.DataFrame(data=self.c.fetchall(), columns=['tid', 'tag'])

    def fetch_all_tids_tags_threshold(self, threshold = 0):
        ''' Returns a list of tuples containing tids and tags (as they appear in the tid_tag table) satisfying val > threshold. '''

        q = "SELECT tid, tag FROM tid_tag WHERE val > ?"
        self._query(q, threshold)
        return pd.DataFrame(data=self.c.fetchall(), columns=['tid', 'tag'])

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
        tids: list
            List containing tids as strings.

        Returns
        -------
        tag_dict: dict
            The keys are the tids from the input list.
            The values are lists of tags for each given tid.
        '''

        tags_dict = {}
        for tid in tids:
            tags_dict[tid] = self._query_tags(tid)
        return tags_dict

    def tid_tag_count(self, tids):
        ''' Given a list of tids, returns a dict with tids as keys and its number of tags as value.
        
        Parameters
        ----------
        tids: list
            List containing tids as strings.

        Returns
        -------
        count_dict: dict
            The keys are the tids from the input list.
            The values are the number of tags for each given tid.
        '''

        count_dict = {}
        for tid in tids:
            count_dict[tid] = len(self._query_tags(tid))
        return count_dict

    def tid_tag_count_filter(self, tids, min_tags):
        ''' Given a list of tids, filters out those with less than minimum number of tags. 
        
        Parameters
        ----------
        tids: list
            List containing tids as strings.
        
        min_tags: int
            Minimum number of tags to allow in output list.
        '''

        count_dict = self.tid_tag_count(tids)
        tids_filtered = [tid for tid in tids if count_dict[tid] >= min_tags]
        return tids_filtered

    def with_tag(self, tag):
        ''' Return all tids with a given tag. '''
        
        q = "SELECT tid FROM tid_tag WHERE tag = ?"
        tag_num = self.tag_to_tag_num(tag)
        self._query(q, tag_num)
        return [self.tid_num_to_tid(i[0]) for i in self.c.fetchall()]
        
    def popularity(self):
        ''' Produces a dataframe with the following columns: 'tag', 'tag_num', 'count'. '''
        
        q = "SELECT tag, count(tag) FROM tid_tag GROUP BY tag ORDER BY count(tag) DESC"
        self._query(q)
        l = self.c.fetchall() # return list of tuples of the form (tag_num, count)
        
        # add tag to list of tuples
        for i, entry in enumerate(l):
            l[i] = (self.tag_num_to_tag(entry[0]), ) + entry
        
        # create df
        pop = pd.DataFrame(data=l, columns=['tag', 'tag_num', 'count'])
        pop.sort_values('count', ascending=False, inplace=True)
        pop.index += 1
        return pop

class LastFm2Pandas():
    ''' Reads the last.fm database into different pandas dataframes and provides methods to perform advanced queries on it.

    Methods
    -------
    - tid_to_tid_num
        Return tid_num(s) given tid(s).

    - tid_num_to_tid
        Return tid(s) given tid_num(s).

    - tid_num_to_tag_nums
        Return tag_num(s) given tid_num(s).

    - tid_num_to_tags
        Return tag(s) given tid_num(s).

    - tag_num_to_tag
        Return tag(s) given tag_num(s).

    - tag_to_tag_num
        Return tag_num(s) given tag(s).

    - get_tags
        Return a list of all the tags.
        
    - get_tag_nums
        Return a list of all the tag_nums.

    - get_tids
        Get tids which have at least one tag.
        
    - get_tid_nums
        Get tid_num of tids which have at least one tag.

    - query_tags
        Get tags for given tid(s).

    - fetch_all_tids_tags
        Return a dataframe containing tids and tags (as they appear in the tid_tag table).
        
    - fetch_all_tids_tags_threshold
        Return a dataframe containing tids and tags (as they appear in the tid_tag table) satisfying val > threshold.
    
    - with_tag
        Return all tids with a given tag.

    - popularity
        Return a dataframe containing the tags ordered by popularity, together with the number of times they appear.
    '''

    def __init__(self, path, tags=None, no_tags=False, tids=None, no_tids=False, tid_tag=None, no_tid_tag=False):
        '''
        Parameters
        ----------
        path: str
            Path to tags database. Defaults to path on Boden.

        tags: pd.DataFrame
            If path is None, you can directly provide a tags dataframe.

        no_tags: bool
            If True, do not store tags table.

        tids: pd.DataFrame
            If path is None, you can directly provide a tids dataframe.

        no_tids: bool
            If True, do not store tids table.
        
        tid_tag: pd.DataFrame
            If path is None, you can directly provide a tids dataframe.

        no_tid_tag: bool
            If True, do not store tid_tag table.
        '''

        if path is not None:
            if not os.path.isfile(path):
                raise OSError("file " + path + " does not exist!")

            conn = sqlite3.connect(path)
            if not no_tags:
                self.tags = pd.read_sql_query('SELECT * FROM tags', conn)
                self.tags.index += 1
            else:
                self.tags = None
            if not no_tids:
                self.tids = pd.read_sql_query('SELECT * FROM tids', conn)
                self.tids.index += 1
            else:
                self.tids = None
            if not no_tid_tag:
                self.tid_tag = pd.read_sql_query('SELECT * FROM tid_tag', conn)
                self.tid_tag.index += 1
            else:
                self.tid_tag = None
            conn.close()
        else:
            self.tags = tags
            self.tids = tids
            self.tid_tag = tid_tag

    @classmethod
    def load_from(cls, tags=None, tids=None, tid_tag=None): # skip the queue, and load straight from the dataframes
        return cls(path=None, tags=tags, tids=tids, tid_tag=tid_tag)

    def tid_to_tid_num(self, tid, order=False):
        ''' Returns tid_num(s) given tid(s).
        
        Parameters
        ----------
        tid: str, array-like
            A single tid or an array-like structure containing tids
            
        order: bool
            If True, order of output matches order of input when input is array-like


        Returns
        -------
        tid_num: str, pandas.Index
            if tid is a string: 
                corresponding tid_num (int)
            if array_like: 
                pd.Index object containing tid_nums
        '''

        if isinstance(tid, str):
            return int(self.tids.loc[self.tids.tid == tid].index[0])
        
        if order:
            return [self.tids.loc[self.tids.tid==t].index[0] for t in tid]

        return self.tids.loc[self.tids.tid.isin(tid)].index.tolist()

    def tid_num_to_tid(self, tid_num, order=False):
        ''' Returns tid(s) given tid_num(s).
        
        Parameters
        ----------
        tid_num: int, array-like
            A single tid_num or an array-like structure containing tid_nums.
        
        order: bool
            If True, order of output matches order of input when input is array-like

        Returns
        -------
        tid: str, ndarray
            if tid_num is an int: 
                corresponding tid (str)
            if array-like: 
                ndarray containing corresponding tids
        '''

        if np.issubdtype(type(tid_num), np.integer):
            return self.tids.at[tid_num, 'tid']
        
        if order:
            return np.array([self.tids.loc[self.tids.index==num, 'tid'].tolist()[0] for num in tid_num], dtype=object)
        
        return self.tids.loc[self.tids.index.isin(tid_num), 'tid'].values

    def tid_num_to_tag_nums(self, tid_num):
        ''' Returns tag_nums given tid_num(s).
        
        Parameters
        ----------
        tid_num: int, array-like
            A single tid_num or an array-like structure containing tid_nums

        Returns
        -------
        tag_nums: ndarray, series
            if tid_num is an int:
                ndarray of corresponding tag_nums (int)
            if array-like:
                pd.Series where indices are tid_nums and values are
                corresponding list of tag_nums
        '''
        
        if np.issubdtype(type(tid_num), np.integer):
            return self.tid_tag.loc[self.tid_tag.tid == tid_num, 'tag'].values

        tag_nums = [self.tid_tag.loc[self.tid_tag.tid == num, 'tag'].values for num in tid_num]
        return pd.Series(tag_nums, index=tid_num)

    def tid_num_to_tags(self, tid_num):
        ''' Gets tags for given tid_num(s).
        
        Parameters
        ----------
        tid_num: int, array-like
            A single tid_num or an array-like structure containing tid_nums

        Returns
        -------
        tags: ndarray, pd.Series 
            if tid_num is an int:
                ndarray containing corresponding tags     
            if array-like:
                pd.Series having tid_nums as indices and list of tags as values
        '''

        tag_nums = self.tid_num_to_tag_nums(tid_num)

        if isinstance(tag_nums, (list, np.ndarray)):
            return self.tag_num_to_tag(tag_nums)

        return tag_nums.map(self.tag_num_to_tag)

    def tag_num_to_tag(self, tag_num, order=False):
        ''' Returns tag(s) given tag_num(s).

        Parameters
        ----------
        tag_num: int, array-like
            A single tag_num or an array-like structure containing tag_nums
            
        order: bool
            If True, order of output matches order of input when input is array-like


        Returns
        -------
        tag: str, ndarray
            if tid_num is an int:
                corresponding tag (str)
            if array-like:
                ndarray containing corresponding tags
        '''

        if np.issubdtype(type(tag_num), np.integer):
            return self.tags.at[tag_num, 'tag']
        
        if order:
            return np.array([self.tags.loc[self.tags.index==num, 'tag'].tolist()[0] for num in tag_num], dtype=object)
        
        return self.tags.loc[self.tags.index.isin(tag_num), 'tag'].values

    def tag_to_tag_num(self, tag, order=False):
        ''' Returns tag_num(s) given tag(s).

        Parameters
        ----------
        tag: str, array-like
            A single tag or an array-like structure containing tags
            
        order: bool
            If True, order of output matches order of input when input is array-like


        Returns
        -------
        tag_num: int, pd.Index
            if tag is a str:
                Corresponding tag_num (int)
            if array-like:
                pd.Index containing corresponding tag_nums
        '''

        if isinstance(tag, str):
            return int(self.tags.loc[self.tags.tag == tag].index[0])
        
        if order:
            return np.array([self.tags.loc[self.tags.tag==t].index[0] for t in tag])
        
        return self.tags.loc[self.tags.tag.isin(tag)].index.values

    def get_tags(self):
        ''' Returns a list of all the tags. '''

        return self.tags['tag'].tolist()

    def get_tag_nums(self):
        ''' Returns a list of all the tag_nums. '''

        return self.tags.index.tolist()

    def get_tids(self):
        ''' Gets tids which have at least one tag. '''

        return self.tids['tid'][-self.tids['tid'].isna()].tolist()

    def get_tid_nums(self):
        ''' Gets tid_num of tids which have at least one tag. '''

        return self.tids.index[-self.tids['tid'].isna()].tolist()

    def query_tags(self, tid):
        ''' Gets tags for given tid(s) 
        
        Parameters
        ----------
        tid: str, array-like
            A single tid or an array-like structure containing tids

        Returns
        -------
        tags: ndarray, pd.Series
            if tag is a str:
                ndarray containing corresponding tags
            if array-like:
                pd.Series having tids as indices and list of tags as values
        '''

        tags = self.tid_num_to_tags(self.tid_to_tid_num(tid))

        if isinstance(tags, (list, np.ndarray)):
            return tags

        return tags.rename(self.tid_num_to_tid)

    def fetch_all_tids_tags(self):
        ''' Returns a list of tuples containing tids and tags (as they appear in the tid_tag table). '''

        return self.tid_tag[['tid', 'tag']]

    def fetch_all_tids_tags_threshold(self, threshold = 0):
        ''' Returns a list of tuples containing tids and tags (as they appear in the tid_tag table) satisfying val > threshold. '''

        return self.tid_tag[['tid', 'tag']][self.tid_tag['val'] > threshold]

    def with_tag(self, tag):
        ''' Returns all tids with a given tag. '''
        
        tag_idx = self.tag_to_tag_num(tag)
        tids = self.tid_tag['tid'][self.tid_tag['tag'] == tag_idx]
        tids = self.tid_num_to_tid(tids)
        return tids.tolist()

    def popularity(self):
        ''' Produces a dataframe with the following columns: 'tag', 'tag_num', 'count'. '''

        df_1 = self.tid_tag['tag'].value_counts().to_frame()
        df_2 = self.tags['tag'].to_frame()
        self.pop = df_2.merge(df_1, left_index=True, right_index=True)
        self.pop.rename(columns={self.pop.columns[0]:'tag', self.pop.columns[1]:'count'}, inplace=True)
        self.pop.sort_values('count', ascending=False, inplace=True)
        self.pop.reset_index(inplace=True)
        self.pop.rename(columns={'index':'tag_num'}, inplace=True)
        self.pop = pd.concat([self.pop['tag'], self.pop['tag_num'], self.pop['count']], axis=1)
        self.pop.sort_values('count', ascending=False, inplace=True)
        self.pop.index += 1
        return self.pop

class Matrix():
    ''' Reads the last.fm database from either LastFm2Pandas (or LastFm, but it would be slower). Provides methods to perform numerical analyses of how tags are distributed among tracks. 
    
    Methods
    -------
    - matrix
        Compute a n-dimensional matrix, where the (i_1, ..., i_n)-th entry represents the number of tracks with all the i-th's tags.
    
    - matrix_load
        Load a pre-computed matrix (saved as a .npz and a .nfo file).

    - tags_and
        Return the number of tracks which have the first tag, AND the second tag, ..., AND the n-th tag.

    - tags_or
        Return the number of tracks which have the first tag, OR the second tag, ..., OR the n-th tag.

    - with_one_without_many
        Compute the number of tracks which have one tag, but not any of some others.

    - with_many_without_one
        Compute the number of tracks which have all of some tags, but not one other.

    - correlation_matrix_2d
        Produce a 2-dimensional matrix showing how 2 tags are correlated.

    - correlation_matrix_3d
        Produce a 3-dimensional matrix showing how 3 tags are correlated.

    - correlation_plot
        Plot a correlation matrix.

    - are_equivalent
        Analyze which tags are arguably equivalent.

    - all_tag_is
        Analyze which tags are for the most part contained into another.

    - all_tag_is_either
        Analyze which tags are for the most part contained into the union of other two tags.
    '''

    def __init__(self, lastfm, tags, dim=3, save_to=None, load_from=None):
        '''
        Parameters
        ----------
        lastfm: LastFm, LastFm2Pandas
            Instance of tags database. Using LastFm2Pandas is strongly recommended here.

        tags: list
            List of tags to use. If None, all the tags will be used.

        save_to: str
            Filename or full path of the .npz file to save matrix and matrix tags. Use to load_from in the future.
        
        dim: int
            The dimension of the matrix.

        load_from: str
            Filename or full path of the .npz file to load matrix and matrix tags.
        '''
        
        if load_from is None:
            self.m, self.m_tags = self.matrix(lastfm, tags=tags, dim=dim, save_to=save_to)
        else:
            self.m, self.m_tags = self.matrix_load(load_from)
        
    @classmethod
    def load_from(cls, path): # skip the queue, and load from a previously saved file
        return cls(None, None, load_from=path)

    def matrix(self, lastfm, tags=None, dim=3, save_to=None):
        ''' Computes a n-dimensional matrix where the (i_1, ... ,i_n)-th entry contains the number of tracks having all the i_1-th, ..., i_n-th tags (where the i's are the indexes in self.m_tags).

        Notes
        -----
        To optimize performance, values are computed only with indexes in increasing order (which means, we only compute the number of tracks having tag-0 
        and tag-1, not vice-versa). This is something to keep in mind when indexing the matrix.
        
        To optimize memory, the matrix is saved in sparse format. DOK is the preferred sparse format for building and indexing, while COO is the preferred
        sparse format to perform mathematical operations).

        The dimension of the matrix captures the kind of queries which you will be able to perform. A matrix of dim=2 on tracks=['rock', 'pop', 'hip-hop'] will
        capture how many tracks have tags rock and pop, or pop and hip-hop, but not rock, pop and hip-hop at the same time.
        
        A matrix of dim=len(tags) will fully describe the database (or the subset of the database having the given tags).
        A matrix of dim>len(tags) will be rather pointless (but we won't prevent you from doing it).

        Parameters
        ----------
        lastfm: LastFm, LastFm2Pandas
            Instance of tags database. Using LastFm2Pandas is strongly recommended here.

        tags: list
            List of tags to use. If None, all the tags will be used.

        dim: int
            The dimension of the matrix.

        save_to: str
            Filename or full path of the .npz file to save matrix and matrix tags. Use to load_from in the future.
        '''

        # initialize matrix tags
        if tags is None:
            tags = lastfm.get_tags()
        else:
            tags = [tag for tag in tags if tag in lastfm.get_tags()] # possibly purge inexistent tags
        
        # initialize matrix
        matrix = sparse.DOK((len(tags), )*dim, dtype=np.int32) # sparse dict-of-keys matrix (for easy creation, awful for calculations)

        # compute total number of steps to comatplotlibetion (see http://www.iosrjournals.org/iosr-jm/papers/Vol8-issue3/A0830110.pdf)
        n_steps = crazysum(n=len(tags), s=3, k=dim-1)

        # check whether a progress bar is needed
        verbose = n_steps > 100
        if verbose:
            progbar = Progbar(n_steps) # instantiate progress bar
        
        def count_intersect_tags(tags):
            tids_list = [lastfm.with_tag(tag) for tag in tags]
            tids_list.sort(key=len, reverse=True)
            tids = set(tids_list.pop()) # start with shortest list of tids to improve performance; convert to set to be able to intersect
            for _ in range(len(tids_list)):
                tids = tids.intersection(tids_list.pop()) # intersections performed from shortest list to longest
            return len(tids) # how many tids have all tags
        
        def count_intersect_tags_recursive(tags_idxs, dim): # recursively iterate count_intersect_tags dim times; avoid repetitions such as 'rock AND pop AND folk' vs. 'rock AND folk AND pop' vs. 'folk AND pop AND rock'
            if dim>=1:
                for i in range(tags_idxs[-1] + 1):
                    count_intersect_tags_recursive(tags_idxs + (i, ), dim-1)
            else:
                matrix[tags_idxs] = count_intersect_tags(np.take(tags, tags_idxs)) # add count to sparse matrix
                if verbose:
                    progbar.add(1)
        
        # instantiate recursive loop
        for i in range(len(tags)):
            count_intersect_tags_recursive((i, ), dim-1)
        
        matrix = matrix.to_coo() # convert to coordinate matrix
        
        if save_to is not None:
            # save matrix
            sparse.save_npz(save_to, matrix.to_coo()) # default to compressed format (i.e. sparse format)

            # save matrix tags in serialized format
            with open(os.path.splitext(save_to)[0] + '.nfo', 'wb') as f:
                pickle.dump(tags, f)
        
        return matrix, tags

    def matrix_load(self, path):
        ''' Loads a previously saved matrix from a .npz file (containing the matrix) and a .nfo file (containing the matrix tags). '''

        # load matrix
        matrix = sparse.load_npz(os.path.splitext(path)[0] + '.npz')
        matrix = sparse.DOK(matrix) # convert to dict-of-keys for faster indexing

        # load matrix tags
        with open (os.path.splitext(path)[0] + '.nfo', 'rb') as f:
            tags = pickle.load(f)

        return matrix, tags

    def tags_and(self, tags):
        ''' Computes how many tracks have all the tags in 'tags'. Provides an easier way to index the matrix.
        
        Parameters
        ----------
        tags: list 
            List of tags.
        '''

        if isinstance(tags, (str, int)):
            tags = [tags]

        tags = np.array(list(set(tags))) # remove duplicates; convert to np.ndarray
        
        assert len(tags) <= len(self.m.shape), 'too many tags provided; try with a matrix of higher dimension'
        
        if tags.dtype == int:
            idxs = tags
        else:
            idxs = np.where(np.array([self.m_tags] * len(tags)) == tags[:,None])[1] # 'numpy version' of idxs = [self.m_tags.index(tag) for tag in tags]
            assert len(idxs) == len(tags), 'some of the tags you inserted might not exist in the matrix' # sanity check (np.where will not return errors if correspondent idx does not exist)
            
        idxs = sorted(idxs, reverse=True) # matrix is sparse, idxs needs to be ordered
        idxs.extend([idxs[-1]] * (len(self.m.shape)-len(idxs))) # match matrix shape if less idxs are provided
        return self.m[tuple(idxs)]

    def tags_or(self, tags):
        ''' Computes how many tracks have at least one of the tags in 'tags'.
        
        Parameters
        ----------
        tags: list 
            List of tags.
        '''

        if isinstance(tags, (str, int)):
            tags = [tags]

        tags = np.array(list(set(tags))) # remove duplicates; convert to np.ndarray
        
        assert len(tags) <= len(self.m.shape), 'too many tags provided; try with a matrix of higher dimension'
        
        if tags.dtype == int:
            idxs = tags
        else:
            idxs = np.where(np.array([self.m_tags] * len(tags)) == tags[:,None])[1] # 'numpy version' of idxs = [self.m_tags.index(tag) for tag in tags]
            assert len(idxs) == len(tags), 'some of the tags you inserted might not exist in the matrix' # sanity check (np.where will not return errors if correspondent idx does not exist)
        
        return sum((-1)**(i+1) * self.tags_and(subset) # inclusion-exclusion principle
                for i in range(1, len(idxs) + 1)
                for subset in itertools.combinations(idxs, i))

    def with_one_without_many(self, with_tags, without_tags):
        ''' Computes how many tracks have at the tag 'with_tags', but not any of the tags in 'without_tags'.
        
        Parameters
        ----------
        with_tags: list 
            List of tags the tracks in the output list will have.
        
        without_tags: list 
            List of tags the tracks in the output list will not have.
        '''

        if isinstance(with_tags, (str, int)):
            with_tags = [with_tags]

        assert len(with_tags) == 1 and len(without_tags) >= 1
        return self.tags_or(with_tags + without_tags) - self.tags_or(without_tags)

    def with_many_without_one(self, with_tags, without_tags): # when with_one_without_one is needed, this function should be preferred
        ''' Computes how many tracks have all the tags in 'with_tags', but not the tag 'without_tags'.
        
        Parameters
        ----------
        with_tags: list 
            List of tags the tracks in the output list will have.
        
        without_tags: list 
            List of tags the tracks in the output list will not have.
        '''

        if isinstance(without_tags, (str, int)):
            without_tags = [without_tags]

        assert len(with_tags) >= 1 and len(without_tags) == 1
        return self.tags_and(with_tags) - self.tags_and(with_tags + without_tags)

    def correlation_matrix_2d(self, plot=False, save_plot_to=None):
        ''' Returns a 2-dimensional matrix whose values indicate the correlation between 2 tags. 
        
        Notes
        -----
        Each i,j-th entry indicates the percentage of tracks with the i-th tag which ALSO have the j-th tag.
        If there are 40.000 'alternative' tracks, and 30.000 of those are also 'rock', assuming that
        'alternative' is the 2-nd tag and 'rock' is the 4-th tag within self.m_tags, then correlation_matrix[2,4] will be 0.75.

        Parameters
        ----------
        plot: bool 
            If True, display the correlation matrix graphically.
        
        save_plot_to: str
            If not None, save plot to the specified path. The format is inferred by the file extension. 
        '''

        l = len(self.m_tags)
        
        assert l >= 2, 'you need to have at least a 2-dimensional matrix'
        
        matrix = np.zeros((l, )*2) # initialize output matrix
        
        for i in range(l):
            for j in range(l):
                tot = self.tags_and([i])
                matrix[i,j] = 1 - (self.with_many_without_one(with_tags=[i], without_tags=[j]) / tot)
        
        if plot:
            plt_matrix = np.copy(matrix)
            np.fill_diagonal(plt_matrix, 0)
            self.correlation_plot(plt_matrix, save_to=save_plot_to) # corrrelation of tag with itself is always 1
        
        return matrix

    def correlation_matrix_3d(self, plot=False, save_plot_to=None):
        ''' Returns a 3-dimensional matrix whose values indicate the correlation between 3 tags. 
        
        Notes
        -----
        Each i,j,k-th entry indicates the percentage of tracks with the i-th tag which ALSO have either the j-th tag or the k-th tag.
        If there are 40.000 'alternative' tracks, and 30.000 of those are either 'rock' or 'alternative rock', assuming that
        'alternative' is the 2-nd tag, 'rock' is the 4-th tag and 'alternative rock' is the 5-th tag within self.m_tags, then correlation_matrix[2,4,5] will be 0.75.
        
        Parameters
        ----------
        plot: bool, int
            If True, display the correlation matrix graphically for all tags. 
            If not False, an int might be specified. In that case, only the correlation for the i-th tag will be displayed.
            If False, do nothing.
        
        save_plot_to: str
            If not None, save plot to the specified path. The format is inferred by the file extension. 
        '''

        l = len(self.m_tags)
        
        assert l >= 3, 'you need to have at least a 3-dimensional matrix'
        
        matrix = np.zeros((l, )*3) # initialize output matrix
        
        for i in range(l):
            for j in range(l):
                for k in range(l):
                    tot = self.tags_and([i])
                    matrix[i,j,k] = 1 - (self.with_one_without_many(with_tags=[i], without_tags=[j,k]) / tot)
        
        if plot is not False:
            def get_plt_matrix(self, matrix): 
                plt_matrix = np.copy(matrix[idx,:,:])
                plt_matrix = np.delete(plt_matrix, idx, 0)
                plt_matrix = np.delete(plt_matrix, idx, 1) # corrrelation of tag with itself is always 1
                tags = np.delete(self.m_tags, idx)
                return plt_matrix, tags

            # plot correlation for idx-th tag only
            if plot is not True:
                assert isinstance(plot, int), 'plot must have type either bool or int'
                idx = plot
                tag = self.m_tags[idx]
                self.correlation_plot(*get_plt_matrix(self, matrix), title=tag, save_to=save_plot_to)

            # plot correlation for all
            else:
                for idx in range(len(self.m_tags)):
                    tag = self.m_tags[idx]
                    self.correlation_plot(*get_plt_matrix(self, matrix), title=tag)
        
        return matrix

    def correlation_plot(self, correlation_matrix, tags=None, title=None, save_to=None):
        ''' Plots a 2-dimensional correlation matrix graphically. 
        
        Parameters
        ----------
        correlation_matrix: np.ndarray
            The 2-dimensional correlation matrix to plot.
        
        tags: list
            If not None, use this list of tags instead of self.m_tags. Its length must be compatible
            with the shape of correlation_matrix. Useful when removing 'trivial entries' (that is, where correlation of tag
            with itself is 1) from the original matrix.
        
        title: str
            If not None, display title.
        
        save_to: str
            If not None, save plot to the specified path. The format is inferred by the file extension. 
        '''
        
        if tags is not None:
            assert len(tags) == correlation_matrix.shape[0] # check whether a valid list of tags has been provided
        else:
            tags = self.m_tags

        res = correlation_matrix.shape[0]//10
        fig, ax = plt.subplots(figsize=[res,res])
        im = ax.imshow(correlation_matrix, cmap=matplotlib.cm.Blues)
        ax.set_xticks(np.arange(len(tags)))
        ax.set_yticks(np.arange(len(tags)))
        ax.set_xticklabels(tags, rotation='vertical')
        ax.set_yticklabels(tags)
        ax.set_aspect('auto')

        if title:
            plt.title(title,fontweight="bold")

        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel('correlation')
        plt.show()

        if save_to:
            assert os.path.splitext(save_to)[1] != '', 'please enter a valid path'
            plt.savefig(save_to)

    def are_equivalent(self, threshold=0.8, verbose=False):
        ''' Reads the 2-dimensional correlation matrix to present a human-readable outline of the tags which are arguably equivalent.

        Parameters
        ----------
        threshold: float
            The proportion of correlation above which two tags are considered equivalent.
        
        verbose: bool
            If True, interpret the matrix nicely.
        '''

        correlation = self.correlation_matrix_2d()
        matrix = np.where((correlation > threshold) & np.transpose(correlation > threshold), correlation, 0)
        np.fill_diagonal(matrix, 0)
        if verbose:
            count = 0
            for x, y in zip(*np.where(np.tril(matrix>0))):
                count+=1
                print('{0:>3}. {1:3.1f}% of {3} is {4}\n{5:>3}  {2:3.1f}% of {4} is {3}\n'.format(count, correlation[x,y]*100, correlation[y,x]*100, self.m_tags[x], self.m_tags[y], ' ' * len(str(count))))
        return matrix

    def all_tag_is(self, threshold=0.7, verbose=False):
        ''' Reads the 2-dimensional correlation matrix to present a human-readable outline of the tags which are for the most part sub-tags of another.
        
        Parameters
        ----------
        threshold: float
            The proportion of correlation above which two tags are considered one a subset of the other.
        
        verbose: bool
            If True, interpret the matrix nicely.
        '''

        correlation = self.correlation_matrix_2d()
        matrix = np.where((correlation > threshold), correlation, 0)
        np.fill_diagonal(matrix, 0)
        if verbose:
            count = 0
            for x, y in zip(*np.where(matrix>0)):
                count+=1
                print('{0:>3}. {1:3.1f}% of {3} is {4}\n{5}(but {2:3.1f}% of {4} is {3})\n'.format(count, correlation[x,y]*100, correlation[y,x]*100, self.m_tags[x], self.m_tags[y], ' ' * max(0, len(str(count))-3)))
        return matrix

    def all_tag_is_either(self, threshold=0.7, verbose=False):
        ''' Reads the 3-dimensional correlation matrix to present a human-readable outline of the tags which are for the most part contained in the union of two other tags.
        
        Parameters
        ----------
        threshold: float
            The proportion of correlation above which two tags are considered one a subset of the union of the other two.
        
        verbose: bool
            If True, interpret the matrix nicely.
        '''

        correlation = self.correlation_matrix_3d()
        matrix = np.where((correlation > threshold), correlation, 0)
        for i in range(len(self.m_tags)):
            matrix[i,i,:]=0
            matrix[i,:,i]=0
        if verbose:
            count = 0
            for x, y, z in zip(*np.where(matrix>0)):
                if y == z:
                    continue # this values are captured by all_tag_is()
                count+=1
                print('{0:>3}. {1:3.1f}% of {2} is either {3}\n{5}or {4}\n'.format(count, correlation[x,y,z]*100, self.m_tags[x], self.m_tags[y], self.m_tags[z], ' ' * (22 + len(self.m_tags[x]))))
        return matrix
    
def crazysum(n, s, k):
    return int((math.factorial(n+k-1)/(math.factorial(n-1)*math.factorial(k+1)))*((n-1)*s+k+3-2*n))
