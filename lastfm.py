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
    Open a connection to the db file and provide methods to perform queries on it.
    This class is faster to init, but some queries (expecially on the tid_tag table) might take some time to perform.

- LastFm2Pandas
    Read the database into three pandas dataframes and prodive methods to retrieve information from them.
    This class is slower to init, since the whole database is loaded into memory, but consequently queries are much faster. This class also contain some additional "advanced" methods.
'''

import itertools
import math
import os
import pickle
import sqlite3

import pandas as pd
import numpy as np
import sparse

from utils import MyProgbar

DEFAULT = '/srv/data/msd/lastfm/SQLITE/lastfm_tags.db'

class LastFm():
    ''' Opens a SQLite connection to the last.fm database. Provides methods to perform advanced queries on it.

    Methods
    -------
    - tid_to_tid_num
        Get tid_num given tid.

    - tid_num_to_tid
        Get tid given tid_num.

    - tid_num_to_tag_nums
        Get tag_num given tid_num.

    - tag_num_to_tag
        Get tag given tag_num.

    - tag_to_tag_num
        Get tag_num given tag.

    - vec_tag_num_to_tag
        Get tag given tag_num (vectorized version).

    - vec_tag_to_tag_num
        Get tag_num given tag (vectorized version).

    - vec_tid_to_tid_num
        Get tid_num given tid (vectorized version).

    - vec_tid_num_to_tid
        Get tid given tid_num (vectorized version).

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

    - fetch_all_tids_tags
        Return a dataframe containing tids and tags (as they appear in the tid_tag table).
        
    - fetch_all_tids_tags_threshold
        Return a dataframe containing tids and tags (as they appear in the tid_tag table) satisfying val > threshold.

    - tid_tag_count
        Given a list of tids, returns a dict with tids as keys and its number of tags as value.

    - tid_tag_count_filter
        Given a list of tids, filters out those with less than minimum number of tags.
        
    - db_to_csv
        Convert the tags database into three different .csv files.

    - popularity
        Return a dataframe containing the tags ordered by popularity, together with the number of times they appear.
    '''

    def __init__(self, path):
        if not os.path.isfile(path):
            raise OSError("file " + path + " does not exist!")

        self.conn = sqlite3.connect(path)
        self.c = self.conn.cursor()
        self.path = path

        self.vec_tag_num_to_tag = np.vectorize(self.tag_num_to_tag)
        self.vec_tag_to_tag_num = np.vectorize(self.tag_to_tag_num)
        self.vec_tid_num_to_tid = np.vectorize(self.tid_num_to_tid)
        self.vec_tid_to_tid_num = np.vectorize(self.tid_to_tid_num)

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
        return pd.DataFrame(data=self.c.fetchall(), columns=['tid', 'tag'])

    def fetch_all_tids_tags_threshold(self, threshold = 0):
        ''' Returns a list of tuples containing tids and tags (as they appear in the tid_tag table) satisfying val > threshold. '''

        q = "SELECT tid, tag FROM tid_tag WHERE val > ?"
        self.query(q, threshold)
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
            tags_dict[tid] = self.query_tags(tid)
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
            count_dict[tid] = len(self.query_tags(tid))
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

    def db_to_csv(self, output_dir=None):
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
        self.query(q)
        tables = [i[0] for i in self.c.fetchall()]
        for table in tables:
            print('saving '+ 'lastfm' + '_' + table +'.csv')
            path = os.path.join(output_dir, 'lastfm' + '_' + table +'.csv')
            df = pd.read_sql_query("SELECT * FROM " + table, self.conn)
            df.to_csv(path, index_label=False)
        print('Done')
        
    def popularity(self):
        ''' Produces a dataframe with the following columns: 'tag', 'tag_num', 'count'. '''
        
        q = "SELECT tag, count(tag) FROM tid_tag GROUP BY tag ORDER BY count(tag) DESC"
        self.query(q)
        l = self.c.fetchall() # return list of tuples of the form (tag_num, count)
        
        # add tag to list of tuples
        for i, entry in enumerate(l):
            l[i] = (self.tag_num_to_tag(entry[0]), ) + entry
        
        # create df
        pop = pd.DataFrame(data=l, columns=['tag', 'tag_num', 'count'])
        pop.index += 1
        return pop

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

    def __init__(self, path, no_tags=False, no_tids=False, no_tid_tag=False):
        '''
        Parameters
        ----------
        path: str
            Path to tags database. Defaults to path on Boden.

        no_tags: bool
            If True, do not store tags table.

        no_tids: bool
            If True, do not store tids table.

        no_tid_tag: bool
            If True, do not store tid_tag table.
        '''

        if not os.path.isfile(path):
            raise OSError("file " + path + " does not exist!")

        conn = sqlite3.connect(path)
        if not no_tags:
            self.tags = pd.read_sql_query('SELECT * FROM tags', conn)
            self.tags.index += 1
        if not no_tids:
            self.tids = pd.read_sql_query('SELECT * FROM tids', conn)
            self.tids.index += 1
        if not no_tid_tag:
            self.tid_tag = pd.read_sql_query('SELECT * FROM tid_tag', conn)
            self.tid_tag.index += 1
        conn.close()

    def tid_to_tid_num(self, tid, order=False):
        ''' Returns tid_num(s) given tid(s)
        
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
        ''' Returns tid(s) given tid_num(s)
        
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

        if isinstance(tid_num, int):
            return self.tids.at[tid_num, 'tid']
        
        if order:
            return np.array([self.tids.loc[self.tids.index==num, 'tid'].tolist()[0] for num in tid_num], dtype=object)
        
        return self.tids.loc[self.tids.index.isin(tid_num), 'tid'].values

    def tid_num_to_tag_nums(self, tid_num):
        ''' Returns tag_nums given tid_num(s)
        
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
        
        if isinstance(tid_num, int):
            return self.tid_tag.loc[self.tid_tag.tid == tid_num, 'tag'].values

        tag_nums = [self.tid_tag.loc[self.tid_tag.tid == num, 'tag'].values for num in tid_num]
        return pd.Series(tag_nums, index=tid_num)

    def tid_num_to_tags(self, tid_num):
        ''' Gets tags for given tid_num(s) 
        
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
        ''' Returns tag(s) given tag_num(s) 

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

        if isinstance(tag_num, int):
            return self.tags.at[tag_num, 'tag']
        
        if order:
            return np.array([self.tags.loc[self.tags.index==num, 'tag'].tolist()[0] for num in tag_num], dtype=object)
        
        return self.tags.loc[self.tags.index.isin(tag_num), 'tag'].values

    def tag_to_tag_num(self, tag, order=False):
        ''' Returns tag_num(s) given tag(s)

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
        ''' Return all tids with a given tag. '''
        
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
        self.pop.index += 1
        return self.pop

class Matrix():
    def __init__(self, lastfm, tags, dim=3, save_to=None, load_from=None):
        if load_from is None:
            self.m, self.m_tags = self.matrix(lastfm, tags=tags, dim=dim, save_to=save_to)
        else:
            self.m, self.m_tags = self.matrix_load(load_from)
        
    @classmethod
    def load_from(cls, path):
        return cls(None, None, load_from=path)

    def matrix(self, lastfm, tags=None, dim=3, save_to=None):
        # initialize matrix tags
        if tags is None:
            tags = lastfm.get_tags()
        else:
            tags = [tag for tag in tags if tag in lastfm.get_tags()] # possibly purge inexistent tags
        
        # initialize matrix
        matrix = sparse.DOK((len(tags), )*dim, dtype=np.int32) # sparse dict-of-keys matrix (for easy creation, awful for calculations)

        # compute total number of steps to completion (see http://www.iosrjournals.org/iosr-jm/papers/Vol8-issue3/A0830110.pdf)
        n_steps = crazysum(n=len(tags), s=3, k=dim-1)

        # check whether a progress bar is needed
        verbose = n_steps > 5000
        if verbose:
            progbar = MyProgbar(n_steps) # instantiate progress bar
        
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
            sparse.save_npz(save_to, matrix) # default to compressed format (i.e. sparse format)

            # save matrix tags
            with open(os.path.splitext(save_to) + '.nfo', 'wb') as f:
                pickle.dump(tags, f)
        
        return matrix, tags

    def matrix_load(path):
        # load matrix
        matrix = sparse.load_npz(os.path.splitext(path) + '.npz')

        # load matrix tags
        with open (os.path.splitext(path) + '.nfo', 'rb') as f:
            tags = pickle.load(f)

        return matrix, tags

    def tags_et(self, tags):
    
        tags = np.array(list(set(tags))) # remove duplicates; convert to np.ndarray
        
        assert len(tags) <= len(self.m.shape)
        
        if tags.dtype == int:
            idxs = tags
        else:
            idxs = np.where(np.array([self.m_tags] * len(tags)) == tags[:,None])[1] # 'numpy version' of idxs = [self.m_tags.index(tag) for tag in tags]
            assert len(idxs) == len(tags) # sanity check (np.where will not return errors if correspondent idx does not exist)
            
        idxs = sorted(idxs, reverse=True) # matrix is sparse, idxs needs to be ordered
        idxs.extend([idxs[-1]] * (len(self.m.shape)-len(idxs))) # match matrix shape if less idxs are provided
        return self.m[tuple(idxs)]

    def tags_or(self, tags):
        
        tags = np.array(list(set(tags))) # remove duplicates; convert to np.ndarray
        
        assert len(tags) <= len(self.m.shape)
        
        if tags.dtype == int:
            idxs = tags
        else:
            idxs = np.where(np.array([self.m_tags] * len(tags)) == tags[:,None])[1] # 'numpy version' of idxs = [self.m_tags.index(tag) for tag in tags]
            assert len(idxs) == len(tags) # sanity check (np.where will not return errors if correspondent idx does not exist)
        
        return sum((-1)**(i+1) * self.tags_et(subset) # inclusion-exclusion principle
                for i in range(1, len(idxs) + 1)
                for subset in itertools.combinations(idxs, i))

    def with_one_without_many(self, with_tags, without_tags):
        assert len(with_tags) == 1 and len(without_tags) >= 1
        return self.tags_or(with_tags + without_tags) - self.tags_or(without_tags)

    def with_many_without_one(self, with_tags, without_tags): # when with_one_without_one is needed, this function should be preferred
        assert len(with_tags) >= 1 and len(without_tags) == 1
        return self.tags_et(with_tags) - self.tags_et(with_tags + without_tags)

    def correlation_matrix_2d(self):
        l = len(self.m_tags)
        matrix = np.zeros((l, )*2)
        for i in range(l):
            for j in range(l):
                tot = self.tags_et([i])
                matrix[i,j] = 1 - (self.with_many_without_one(with_tags=[i], without_tags=[j]) / tot)
        return matrix

    def correlation_matrix_3d_12(self):
        l = len(self.m_tags)
        matrix = np.zeros((l, )*3)
        for i in range(l):
            for j in range(l):
                for k in range(l):
                    tot = self.tags_et([i])
                    matrix[i,j,k] = 1 - (self.with_one_without_many(with_tags=[i], without_tags=[j,k]) / tot)
        return matrix

    def correlation_matrix_3d_21(self):
        l = len(self.m_tags)
        matrix = np.zeros((l, )*3)
        for i in range(l):
            for j in range(l):
                for k in range(l):
                    tot = self.tags_et([i,j])
                    matrix[i,j,k] = 1 - (self.with_many_without_one(with_tags=[i,j], without_tags=[k]) / tot)
        return matrix

def crazysum(n, s, k):
    return int((math.factorial(n+k-1)/(math.factorial(n-1)*math.factorial(k+1)))*((n-1)*s+k+3-2*n))