''' Contains tool to generate a new lastfm_tags.db file containing only the "clean" merged tags


Notes
-----
This file can be run as a script. To do so, just type 'python clean_lastfm.py' in the terminal. The help 
page should contain all the options you might possibly need. 

The script relies on the query_lastfm module to perform queries on the original database, and on the
lastfm_tool module to obatain a dataframe with the tags to retain and the tags to merge.

The script will output a db file similar in structure to the original lastfm_tags.db. The query_lastfm
module will work on the new db file as well.
'''

import argparse
import ast
import os
import sqlite3
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../modules')))

import query_lastfm as db
import lastfm_tool as lf

def flatten(df: pd.DataFrame):
    '''
    Suppose the tags dataframe looks like this:
    
            tags            merge-tags
    1       rock            ['ROCK']
    2       pop             []
    3       hip hip         ['hiphop', 'hip-hop']

    Then this function returns:
            tags            new_tag_num
    1       rock            1
    2       ROCK            1
    3       pop             2
    4       hip hop         3
    5       hiphop          3
    6       hip-hop         3
    '''

    tags = [] 
    tags_nums = []
    
    for num, row in df.iterrows():
        tag, merge_tags = row
        tags.append(tag)
        tags += merge_tags
        tags_nums += list(np.full(len(merge_tags)+1, num))

    output = pd.DataFrame(data={'tag': tags, 'new_tag_num': tags_nums})
    return output

def flatten_to_tag_num(db: db.LastFm, df: pd.DataFrame):
    '''
    Same as flatten, but:
    1) Now use tag number (from original lastfm db) instead of string (for instance, has 95 instead of 'pop')
    2) Add one line of zeros at the top (...will be used in tag_tag function)
    '''

    output = flatten(df)
    output['tag'] = output['tag'].map(db.tag_to_tag_num)
    output.columns = ['tag', 'new_tag']

    # append row of 0's at the top
    nul = pd.DataFrame(data={'tag': [0], 'new_tag': [0]})
    output.index = output.index + 1
    output = output.append(nul, verify_integrity=True).sort_index()
    return output

def create_tag_tag_table(db: db.LastFm, df: pd.DataFrame):
    '''
    Create a dataframe with two columns: 'old_lastfm_tag' contains all the tag nums from the original lastfm database,
    'new_tag' contains the correspondent 'clean' tag num (that is, the row index in the dataframe produced by Aden). If we
    don't have a correspondent tag (that means that the tag falls below the threshold) use 'new_tag' = 0
    '''
    
    def locate_with_except(series):
        def inner(i):
            try:
                return series.loc[i]
            except KeyError:
                return 0
        return inner

    flat = flatten_to_tag_num(db, df)
    flat['index'] = flat.index.to_series()
    flat = flat.set_index('tag', verify_integrity=True).sort_index()
    
    # fetch the tag num's from the original database
    tag_tag = pd.Series(db.get_tag_nums())
    tag_tag.index = tag_tag.index + 1

    # define a new 'loc_except' function that locates an entry if it exists, and returns 0 otherwise
    flat['new_tag'].loc_except = locate_with_except(flat['new_tag'])

    # for each tag num, get the corresponding 'clean' tag num from the flattened dataframe (returns tag num = 0 if tag falls below the pop threshold)
    tag_tag = tag_tag.map(flat['new_tag'].loc_except)
    return tag_tag

def create_tid_tag_table(db: db.LastFm, tag_tag: pd.DataFrame, tid_tag_threshold: int = None):
    '''
    Create a dataframe with two columns: 'tid' contains all the tid's from the original tid_tag table,
    'tag' contains the new tag. Here all the 0's are dropped.
    '''
    
    # fetch the tids from the original database, and map the tags to their correspondent 'clean' tags
    if tid_tag_threshold is not None:
        tid_tag = db.fetch_all_tids_tags_threshold(tid_tag_threshold)
    else:
        tid_tag = db.fetch_all_tids_tags()
    
    if not isinstance(tid_tag, pd.DataFrame): # type(tid_tag) varies depending on whether db.LastFm or db.LastFm2Pandas is being used
        tids = [tup[0] for tup in tid_tag]
        tags = [tup[1] for tup in tid_tag]
        tid_tag = pd.DataFrame(data={'tid': tids, 'tag': tags}).sort_values('tid')
        tid_tag.reset_index(drop=True, inplace=True)
    
    col_1 = tid_tag['tid']
    col_2 = tid_tag['tag'].map(tag_tag)

    # concatenate columns in a dataframe
    tid_tag = pd.concat([col_1, col_2], axis=1)

    # remove tags which fall below the popularity theshold
    tid_tag = tid_tag[tid_tag['tag'] != 0]
    return tid_tag

if __name__ == "__main__":

    description = "Script to generate a new LastFm database, similar in structure to the original LastFm database, containing only clean the tags for each track."
    epilog = "Example: python clean_lastfm.py ~/lastfm/lastfm_tags.db ~/lastfm/lastfm_tags_clean.db"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("input", help="input db filename or path")
    parser.add_argument("output", help="output db filename or path")
    parser.add_argument('--val-thresh', dest='val', type=float, help="discard tags with val less than threshold")
    
    args = parser.parse_args()
    
    if args.output[-3:] != '.db':
        args.output += '.db'

    if os.path.isfile(args.output):
       print("WARNING file " + args.output + " already exists!")
       sys.exit(0)

    lastfm = db.LastFm(args.input)

    # replace with code to generate dataframe on-the-fly
    debug = False

    if debug == True:
        df = pd.read_csv('tags_temp.csv', usecols=[1,2])
    else:
        df = lf.generate_final_csv()

    assert all(df.columns == ['tag', 'merge_tag'])

    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1
    df['merge_tag'] = df['merge_tag'].map(ast.literal_eval) # without this, lists are parsed are strings

    # generate tables which will go into output database
    tags = df['tag']
    print('Matching all tags to the "clean" few ones...', end=' ', flush=True)
    tag_tag = create_tag_tag_table(lastfm, df)
    print('done')
    print('Matching all tids to tags...', end=' ', flush=True)
    tid_tag = create_tid_tag_table(lastfm, tag_tag, args.val)
    print('done')
    print('Purging tids...', end=' ', flush=True)
    tids = tid_tag['tid'].drop_duplicates()
    tids.index = tids.values
    tids = tids.map(lastfm.tid_num_to_tid).reindex(pd.RangeIndex(1, 505217))
    print('done')

    # generate output
    print('Saving new tables as a db file...', end=' ', flush=True)
    conn = sqlite3.connect(args.output)
    tags.to_sql('tags', conn, index=False)
    tids.to_sql('tids', conn, index=False)
    tid_tag.to_sql('tid_tag', conn, index=False)
    print('done')
    conn.close()
