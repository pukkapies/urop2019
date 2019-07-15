''' Contains simple tools for querying the lastfm_tags.db file


Notes
-----
The lastfm_tags database contains 3 tables: tids, tags, tid_tag.
- tids, 1-column table containing the track ids.
- tid_tags, contains 3 columns:
    - tid: rowid of the track id in the tids table
    - tag: rowid of the tag that belongs to the tid in the same row.
    - val: number between 0 and 100 (guessing this is how accurate the tag is?)
- tags, 1-column table containing the tags.

The row number of the tid in the tids table will be refered to as as tid_num.
Similarly tag_num will refer to the row number of the tag in the tags table.

IMPORTANT: If using this script elsewhere than on boden then run set_path(db_path) to
set the path of the database. Otherwise it will use the default path, which is the path
to the database on boden.


Functions
---------
- set_path              Sets path to the lastfm_tags.db
- tid_to_tid_nums       Gets tid_num given tid
- tid_num_to_tid        Gets tid given tid_num 
- tid_num_to_tag_nums   Gets tag_num given tid_num 
- tag_num_to_tag        Gets tag given tag_num 
- tag_to_tag_num        Gets tag_num given tag 
- get_tags              Gets a list of tags associated to given tid 
- get_tags_dict         Gets a dict with tids as keys and a list of its tags as value
- tid_tag_count         Gets a dict with tids as keys and its number of tags as value 
- filter_tags           Filters list of tids based on minimum number of tags
- tag_count             Gets a dict with the tags associated to tids as keys and their count number as values
'''

import sqlite3

path = '/srv/data/msd/lastfm/SQLITE/lastfm_tags.db' # default path

def set_path(new_path):
    ''' Sets new_path as default path for the lastfm_tags database '''
    global path
    path = new_path

def tid_to_tid_num(tid):
    ''' Returns tid_num, given tid '''

    conn = sqlite3.connect(path)
    q = "SELECT rowid FROM tids WHERE tid ='" + tid + "'"
    res = conn.execute(q)
    output = res.fetchone()[0]
    conn.close()
    return output

def tid_num_to_tid(tid_num):
    ''' Returns tid, given tid_num '''

    conn = sqlite3.connect(path)
    q = "SELECT tid FROM tids WHERE rowid ='" + str(tid_num) + "'"
    res = conn.execute(q)
    output = res.fetchone()[0]
    conn.close()
    return output

def tid_num_to_tag_nums(tid_num):
    ''' Returns list of the associated tag_nums to the given tid_num '''

    conn = sqlite3.connect(path)
    q = "SELECT tag FROM tid_tag WHERE tid ='" + str(tid_num) + "'"
    res = conn.execute(q)
    output = [i[0] for i in res.fetchall()]
    conn.close()
    return output
    
def tag_num_to_tag(tag_num):
    ''' Returns tag given tag_num '''

    conn = sqlite3.connect(path)
    q = "SELECT tag FROM tags WHERE rowid = " + str(tag_num)
    res = conn.execute(q)
    output = res.fetchone()[0]
    conn.close()
    return output

def tag_to_tag_num(tag):
    ''' Returns tag_num given tag '''

    conn = sqlite3.connect(path)
    q = "SELECT rowid FROM tags WHERE tag = " + tag 
    res = conn.execute(q)
    output = res.fetchone()[0]
    conn.close()
    return output

def get_tids_with_tag():
    conn = sqlite3.connect(path)
    q = "SELECT tid FROM tids"
    res = conn.execute(q)
    output = res.fetchall()
    output = [i[0] for i in output]
    conn.close()
    return output

def get_tags(tid):
    ''' Gets tags for a given tid '''
    
    tags = []
    for tag_num in tid_num_to_tag_nums(tid_to_tid_num(tid)):
        tags.append(tag_num_to_tag(tag_num))
    return tags

def get_tags_dict(tids):
    ''' Gets tags for a given list of tids
    
    Parameters
    ----------
    tids : list
        list containing tids as strings

    Returns
    -------
    tag_dict : dict
        - keys are the tids from the tids list
        - values are lists of tags for each given tid
    '''

    tag_dict = {}
    for tid in tids:
        tag_dict[tid] = get_tags(tid)
    return tag_dict

def tag_count(tids):
    ''' Gets number of tags for each given tid 
    
    Parameters
    ----------
    tids : list
        list containing tids as strings

    Returns
    -------
    count_dict : dict
        - keys are the tags associated to any tid from the tids list
        - values are number of tids which the given tag is associated to
    '''

    count_dict = {}
    for tag_list in get_tags_dict(tids).values():
        for tag in tag_list:
            if tag in count_dict:
                count_dict[tag] += 1
            else:
                count_dict[tag] = 1 
    return count_dict

def tid_tag_count(tids):
    ''' Gets number of tags for each given tid 
    
    Parameters
    ----------
    tids : list
        list of tids as strings

    Returns
    -------
    count_dict : dict
        - keys are the tids from the tids list
        - values are number of tags for each given tid
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
