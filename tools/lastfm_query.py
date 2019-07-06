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
'''

import sqlite3

# path_to_lastfm_tags = '/srv/data/msd/lastfm/SQLITE/lastfm_tags.db'
path_to_lastfm_tags = '/home/calle/lastfm_tags.db'


def get_tid_num(tid):
    ''' Returns tid_num, given tid '''

    conn = sqlite3.connect(path_to_lastfm_tags)
    q = "SELECT rowid, tid FROM tids WHERE tid ='" + tid + "'"
    res = conn.execute(q)
    return res.fetchone()[0] 

def get_tag_nums(tid_num):
    ''' Returns list of 1-element tuples with the tag_nums given a tid_num '''

    conn = sqlite3.connect(path_to_lastfm_tags)
    q = "SELECT tag FROM tid_tag WHERE tid ='" + str(tid_num) + "'"
    res = conn.execute(q)
    return res.fetchall()
    
def get_tag(tag_num):
    ''' Returns tag given tag_num '''

    conn = sqlite3.connect(path_to_lastfm_tags)
    q = "SELECT tag FROM tags WHERE rowid = " + str(tag_num)
    res = conn.execute(q)
    return res.fetchone()[0]

def get_tags_from_tids(tids):
    ''' Returns dictionary linking tids to a list containing the tags of that tid '''

    tag_dict = {}
    for tid in tids:
        tag_dict[tid] = [] 
        for tag_num in get_tag_nums(get_tid_num(tid)):
            tag_dict[tid].append(get_tag(tag_num[0]))
    return tag_dict

def tid_tag_count(tids):
    ''' Returns dictionary linking tids to the number of tags ''' 

    count_dict = {}
    tag_dict = get_tags_from_tids(tids)
    for tid in tids:
        count_dict[tid] = len(tag_dict[tid])
    return count_dict

def filter_tags(tids, min_tags):
    ''' Given list of tids, returns list of those with more than min_tags tags '''

    count_dict = tid_tag_count(tids)
    tids_filtered = [tid for tid in tids if count_dict[tid] >= min_tags]
    return tids_filtered

def tag_count(tids):
    ''' Returns a dictionary linking tags to number of occurences among given tids '''

    count_dict = {}
    for tag_list in get_tags_from_tids(tids).values():
        for tag in tag_list:
            if tag in count_dict:
                count_dict[tag] += 1
            else:
                count_dict[tag] = 1 
    return count_dict
