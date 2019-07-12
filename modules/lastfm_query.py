'''
This code contains simple tools for querying the lastfm_tags.db file.


The lastfm_tags tags database contains 3 tables: tids, tags, tid_tag.
- tids, 1-column table containing the track ids.
- tid_tags, contains 3 columns:
    - tid: rowid of the track id in the tids table
    - tag: rowid of the tag that belongs to the tid in the same row.
    - val: number between 0 and 100 (guessing this is how accurate the tag is?)
- tags, 1-column table containing the tags.

In the code I will refer to the row number of the tid in the tids table as tid_num.
Similarly tag_num refers to row number of the tag in the tags table.

Summary of functions:
- set_path:             new_path --> sets path to new_path 
- tid_to_tid_nums:      tid --> tid_num
- tid_num_to_tid:       tid_num --> tid
- tid_num_to_tag_nums:  tid_num --> list of tag_nums
- tag_num_to_tag:       tag_num --> tag
- tag_to_tag_num:       tag --> tag_num
- get_tags:             tid --> list of tags
- get_tags_dict:        tids --> dict with keys: tids, values: list of tags
- tid_tag_count:        tids --> dict with keys: tids, value: number of tags
- filter_tags:          tids, min_tags --> list with tids that have atleast min_tags tags
- tag_count:            tids --> dict with keys: tags, values: number of tids that has this tag
'''

import sqlite3

path = '/srv/data/msd/lastfm/SQLITE/lastfm_tags.db' # default path

def set_path(new_path):
    ''' Sets new_path as path '''
    global path
    path = new_path

def tid_to_tid_num(tid):
    ''' Returns tid_num, given tid '''

    conn = sqlite3.connect(path)
    q = "SELECT rowid FROM tids WHERE tid ='" + tid + "'"
    res = conn.execute(q)
    return res.fetchone()[0] 

def tid_num_to_tid(tid_num):
    ''' Returns tid, given tid_num '''

    conn = sqlite3.connect(path)
    q = "SELECT tid FROM tids WHERE rowid ='" + str(tid_num) + "'"
    res = conn.execute(q)
    return res.fetchone()[0]

def tid_num_to_tag_nums(tid_num):
    ''' Returns list of the tag_nums given a tid_num '''

    conn = sqlite3.connect(path)
    q = "SELECT tag FROM tid_tag WHERE tid ='" + str(tid_num) + "'"
    res = conn.execute(q)
    return [i[0] for i in res.fetchall()]
    
def tag_num_to_tag(tag_num):
    ''' Returns tag given tag_num '''

    conn = sqlite3.connect(path)
    q = "SELECT tag FROM tags WHERE rowid = " + str(tag_num)
    res = conn.execute(q)
    return res.fetchone()[0]

def tag_to_tag_num(tag):
    ''' Returns tag_num given tag '''

    conn = sqlite3.connect(path)
    q = "SELECT rowid FROM tags WHERE tag = " + tag 
    res = conn.execute(q)
    return res.fetchone()[0]

def get_tags(tid):
    ''' Gets tags for a given tid '''
    
    tags = []
    for tag_num in tid_num_to_tag_nums(tid_to_tid_num(tid)):
        tags.append(tag_num_to_tag(tag_num))
    return tags

def get_tags_dict(tids):
    ''' Gets tags for a given list of tids
    
    Input:
    tids -- list of tids

    Output:
    tag_dict -- dictionary
        - keys: tids
        - values: list of tags
    '''

    tag_dict = {}
    for tid in tids:
        tag_dict[tid] = get_tags(tid)
    return tag_dict

def tid_tag_count(tids):
    ''' Gets number of tags for each given tid 
    
    Input:
    tids -- list of tids

    Output:
    tag_dict -- dictionary
        - keys: tids
        - values: number of tags for the given tid
    '''

    count_dict = {}
    for tid in tids:
        count_dict[tid] = len(get_tags(tid))
    return count_dict

def filter_tags(tids, min_tags):
    ''' Given list of tids, returns list of those with more than min_tags tags '''

    count_dict = tid_tag_count(tids)
    tids_filtered = [tid for tid in tids if count_dict[tid] >= min_tags]
    return tids_filtered

def tag_count(tids):
    ''' Gets number of tags for each given tid 
    
    Input:
    tids -- list of tids

    Output:
    tag_dict -- dictionary
        - keys: tags
        - values: number of tids with the given tag 
    '''

    count_dict = {}
    for tag_list in get_tags_dict(tids).values():
        for tag in tag_list:
            if tag in count_dict:
                count_dict[tag] += 1
            else:
                count_dict[tag] = 1 
    return count_dict
