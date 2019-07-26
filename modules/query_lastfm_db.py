'''
'''
import time
import sqlite3

import pandas as pd
import numpy as np





class LastFm():

    def __init__(self, db_path='/srv/data/msd/lastfm/SQLITE/lastfm_tags.db', 
                 no_tags = False, no_tids = False, no_tid_tag = False):
        '''  '''

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
        ''' Returns tid_num, given tid. '''
        if isinstance(tid, str):
            return self.tids.loc[self.tids.tid == tid].index[0]

        return self.tids.loc[self.tids.tid.isin(tid)].index


    def tid_num_to_tid(self, tid_num):
        ''' Returns tid, given tid_num '''

        if isinstance(tid_num, (int, np.integer)):
            return self.tids.at[tid_num, 'tid']

        return self.tids.loc[self.tids.index.isin(tid_num), 'tid'].values

    def tid_num_to_tag_nums(self, tid_num):
        ''' Returns tag_nums, given tid_num '''
        
        # If integer return a list of tag_nums
        if isinstance(tid_num, (int, np.integer)):
            return self.tid_tag.loc[self.tid_tag.tid == tid_num, 'tag'].values
       
        # Else return a series with index tid_num and column tag containing lists of tag_nums
        tag_nums = [self.tid_tag.loc[self.tid_tag.tid == num, 'tag'].values for num in tid_num]
        return pd.Series(tag_nums, index=tid_num)

    def tag_num_to_tag(self, tag_num):
        ''' Returns tag given tag_num '''

        if isinstance(tag_num, (int, np.integer)):
            return self.tags.at[tag_num, 'tag']

        return self.tags.loc[self.tags.index.isin(tag_num), 'tag'].values

    def tag_to_tag_num(self, tag):
        ''' Returns tag_num given tag '''

        if isinstance(tag, str):
            return self.tags.loc[self.tags.tag == tag].index

        return self.tags.loc[self.tags.tag.isin(tag)].index

    def tid_num_to_tags(self, tid_num):
        ''' Gets tags for given tid_num(s) '''

        tag_num = self.tid_num_to_tag_nums(tid_num)

        if isinstance(tag_num, (list, np.ndarray)):
            return self.tag_num_to_tag(tag_num)

        return tag_num.map(self.tag_num_to_tag)

    def tid_to_tags(self, tid):
        ''' Gets tags for given tid(s) '''

        tid_num = self.tid_to_tid_num(tid)
        tags = self.tid_num_to_tags(tid_num)

        if isinstance(tags, (list, np.ndarray)):
            return tags

        return tags.rename(self.tid_num_to_tid)
        


s = time.time()
db = LastFm('/home/calle/lastfm_tags.db')
print(time.time() - s)

# print(db.tid_to_tags(["TRCCOOO128F146368B","TRCCJTI128EF35394A","TRCCHLU128F92FA064","TRCCCYE12903CFF0E9"]))
print(db.tid_to_tags("TRCCOOO128F146368B"))
# print(db.tid_num_to_tags([1, 2, 3, 4, 5]))
