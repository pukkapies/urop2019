''' Contains class which peforms queries on lastfm_tags.db while storing the whole database.

Notes
-----

'''
import time
import sqlite3

import pandas as pd
import numpy as np





class LastFm():
    ''' Class for loading and querying lastfm_tags.db '''

    def __init__(self, db_path='/srv/data/msd/lastfm/SQLITE/lastfm_tags.db', 
                 no_tags = False, no_tids = False, no_tid_tag = False):
        ''' Loads tables from lastfm_dags.db as DataFrames
        
        Parameters
        ----------
        db_path : str
            path to lastfm database. Defaults to path on boden.
        no_tags : bool
            If true do not store tags table
        no_tids : bool
            If true do not store tids table
        no_tid_tag : bool
            If true do not store tid_tag table
        '''

        conn = sqlite3.connect(db_path)

        # Opening tables as databases and shifting index to match rowid in database
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
        # TODO: Should maybe convert index to list?
        # Pros: More flexibility
        # Cons: Not always necessary and will slow peformance
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

        return self.tids.loc[self.tids.tid.isin(tid)].index


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
        
        # If integer return a list of tag_nums
        if isinstance(tid_num, (int, np.integer)):
            return self.tid_tag.loc[self.tid_tag.tid == tid_num, 'tag'].values
       
        # Else return a series with index tid_num and column tag containing lists of tag_nums
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
        ''' Returns tag_num given tag 

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
        


s = time.time()
db = LastFm('/home/calle/lastfm_tags.db', no_tid_tag=True)
print(time.time() - s)

print(db.tag_to_tag_num("blues"))
# print(db.tid_to_tags(["TRCCOOO128F146368B","TRCCJTI128EF35394A","TRCCHLU128F92FA064","TRCCCYE12903CFF0E9"]))
# print(db.tid_to_tags("TRCCOOO128F146368B"))
# print(db.tid_num_to_tags([1, 2, 3, 4, 5]))
