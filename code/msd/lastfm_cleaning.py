''' Contains tools to generate a new lastfm_tags.db file containing only the 'clean' merged tags.


Notes
-----
This file can be run as a script. To do so, just type 'python lastfm_cleaning.py' in the terminal. The help 
page should contain all the options you might possibly need. 

The script relies on the query_lastfm module to perform queries on the original database, and on the
lastfm_tool module to obatain a dataframe with the tags to retain and the tags to merge.

The script will output a .db file similar in structure to the original lastfm_tags.db. The query_lastfm
module will work on the new .db file as well.


Functions
---------
- flatten
    Take the tag dataframe and flatten the tags of the merge_tags column into separate rows.

- flatten_to_tag_num
    Take the tag dataframe and flatten, then replace tag with tag nums from the original database, then add one row of 0's at the top of the output dataframe.

- create_tag_tag_table
    Create a dataframe linking the tag num's in the original lastfm database to the tag num's in the new 'clean' database.

- create_tid_tag_table
    Create a dataframe containing all the 'clean' tags for each tid.

- experimental_clean_add
    Perform advanced cleaning operations on the tid_tag dataframe, and add new tags to it depending on correlation between tags.

- experimental_clean_remove1
    Perform advanced cleaning operations on the tid_tag dataframe, and remove some tags from it depending on correlation between tags.

- experimental_clean_remove2
    Perform advanced cleaning operations on the tid_tag dataframe, and remove some tags from it depending on correlation between tags.
'''

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

import lastfm_cleaning_utils as lastfm_utils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')))

from lastfm import LastFm
from lastfm import LastFm2Pandas

def flatten(df):
    ''' Produce a dataframe with the following columns: 'tag', 'new_tag_num'.

    Parameters
    ----------
    df: pd.DataFrame
        The new tags dataframe to flatten.

    Examples
    --------
    If one entry has tag 'hip-hop' and merge_tags ['hiphop', 'hip hop'], will unpack the tags
    in merge_tags to have one single merge_tag for each row. 
    If such original entry has index 3, will produce the following 4 different 'tag'/'tag_num' entries: 'hip hop' / 3, 'hiphop' / 3, 
    'hiphop' / 3 and 'hip-hop' / 3.
    In other words, the orginal dataframe will be 'flattened out'.
    '''

    tags = [] 
    tags_nums = []
    
    for num, row in df.iterrows():
        tag, merge_tags = row
        tags.append(tag)
        tags += merge_tags
        tags_nums += [num] * (len(merge_tags)+1)

    output = pd.DataFrame(data={'tag': tags, 'new_tag_num': tags_nums})
    return output

def flatten_to_tag_num(fm, df):
    ''' Produce a dataframe with the following columns: 'tag_num', 'new_tag_num'. The tags in the tag column in flatten() are substituted by their original tag nums.
    
    Parameters
    ----------
    fm: LastFm, LastFm2Pandas
        Any instance of the tags database.

    df: pd.DataFrame
        The new tags dataframe to flatten.
    '''

    output = flatten(df)
    output['tag'] = output['tag'].map(fm.tag_to_tag_num)
    output.columns = ['tag_num', 'new_tag_num']
    return output

def create_tag_tag_table(fm, df):
    '''
    Produce a series having the old 'tag_num' as index, as the 'new_tag_num' as values. This will contain all the tags from the original tags database, and 0 as a new tag if the old tag has been discarded.
    
    Parameters
    ----------
    fm: LastFm, LastFm2Pandas
        Any instance of the tags database.

    df: pd.DataFrame
        The new tags dataframe.
    '''
    
    def locate_with_except(series):
        def fn(i):
            try:
                return series.loc[i]
            except KeyError:
                return 0
        return fn

    flat = flatten_to_tag_num(fm, df)
    flat = flat.set_index('tag_num', verify_integrity=True).sort_index()
    
    # define a new 'loc_except' function that locates an entry if it exists, and returns 0 otherwise
    loc_except = locate_with_except(flat['new_tag_num'])

    # for each tag num, get the corresponding 'clean' tag num from the flattened dataframe (returns tag num = 0 if tag falls below the pop threshold)
    tag_tag = pd.Series(fm.get_tag_nums())
    tag_tag.index = tag_tag.values
    tag_tag = tag_tag.map(loc_except)
    return tag_tag

def create_tid_tag_table(fm, tag_tag, tid_tag_threshold = None):
    '''
    Produce a dataframe with the following columns: 'tid', 'tag'. 

    Parameters
    ----------
    fm: LastFm, LastFm2Pandas
        Any instance of the tags database.

    tag_tag: pd.DataFrame
        The tag_tag dataframe produced with create_tag_tag_table().
    
    tid_tag_threshold: int
        The minimum value for val (see original tags database) to allow in the new tid_tag table.
    '''
    
    # fetch the tids from the original database, and map the tags to their correspondent 'clean' tags
    if tid_tag_threshold is not None:
        tid_tag = fm.fetch_all_tids_tags_threshold(tid_tag_threshold)
    else:
        tid_tag = fm.fetch_all_tids_tags()
    
    col_1 = tid_tag['tid']
    col_2 = tid_tag['tag'].map(tag_tag)

    # concatenate columns in a dataframe
    tid_tag = pd.concat([col_1, col_2], axis=1)

    # remove tags which fall below the popularity theshold
    tid_tag = tid_tag[tid_tag['tag'] != 0]
    return tid_tag

def experimental_clean_add(tid_tag, matrix):
    '''
    If the correlation between tag-A and tag-B is higher than a certain threshold (which will determine
    how 'matrix' is generated), add tag-B to all the tracks which have tag-A but not tag-B.

    Parameters
    ----------
    tid_tag: pd.DataFrame
        The tag_tag dataframe produced with create_tid_tag_table().
    
    matrix: np.ndarray
        The two-dimensional correlation matrix produced with lastfm.Matrix.all_tag_is().

    Examples
    --------
    If 'hard rock' is 80% correlated with 'rock', and threshold is less than 80%, add 'rock' to that 20% which has 'hard rock' but not 'rock'.
    '''

    for tag_x, tag_y in zip(*np.where(matrix>0)):
        have_tag_x = tid_tag[tid_tag['tag'] == tag_x+1]
        have_tag_y = tid_tag[tid_tag['tag'] == tag_y+1]
        
        have_both = have_tag_x.reset_index().merge(have_tag_y.reset_index(), on='tid', how='inner')
        
        to_add = have_tag_x.drop(have_both.index_x, inplace=False) # tracks with tag_x, but without tag_y
        to_add['tag'] = tag_y + 1
        
        tid_tag = tid_tag.append(to_add, ignore_index=True, sort=True)
    return tid_tag

def experimental_clean_remove1(tid_tag, matrix):
    '''
    If the correlation between tag-A and tag-B is higher than a certain threshold (which will determine
    how 'matrix' is generated), remove tag-B from the tracks which have both tag-A and tag-B.

    Parameters
    ----------
    tid_tag: pd.DataFrame
        The tag_tag dataframe produced with create_tid_tag_table().
    
    matrix: np.ndarray
        The two-dimensional correlation matrix produced with lastfm.Matrix.all_tag_is().

    Examples
    --------
    If 'hard rock' is 80% correlated with 'rock', and threshold is less than 80%, take that 80% and remove the tag 'rock'.
    '''

    for tag_x, tag_y in zip(*np.where(matrix>0)):
        have_tag_x = tid_tag[tid_tag['tag'] == tag_x+1]
        have_tag_x.reset_index(drop=False, inplace=True)
        have_tag_y = tid_tag[tid_tag['tag'] == tag_y+1]
        have_tag_y.reset_index(drop=False, inplace=True)
        to_drop = have_tag_x.merge(have_tag_y, on='tid', how='inner', suffixes=('_x', '_y'))
        tid_tag.drop(to_drop.index_y, inplace=True)

def experimental_clean_remove2(tid_tag, matrix):
    '''
    If the correlation between tag-A and tag-B is higher than a certain threshold (which will determine
    how 'matrix' is generated), remove tag-A from the tracks which have both tag-A and tag-B.

    Parameters
    ----------
    tid_tag: pd.DataFrame
        The tag_tag dataframe produced with create_tid_tag_table().
    
    matrix: np.ndarray
        The two-dimensional correlation matrix produced with lastfm.Matrix.all_tag_is().

    Examples
    --------
    If 'hard rock' is 80% correlated with 'rock', and threshold is less than 80%, take that 80% and remove the tag 'hard rock', leaving only that 20% of pure 'hard rock'.
    '''
    
    for tag_x, tag_y in zip(*np.where(matrix>0)):
        have_tag_x = tid_tag[tid_tag['tag'] == tag_x+1]
        have_tag_x.reset_index(drop=False, inplace=True)
        have_tag_y = tid_tag[tid_tag['tag'] == tag_y+1]
        have_tag_y.reset_index(drop=False, inplace=True)
        to_drop = have_tag_x.merge(have_tag_y, on='tid', how='inner', suffixes=('_x', '_y'))
        tid_tag.drop(to_drop.index_x, inplace=True)

if __name__ == '__main__':

    description = 'Script to generate a new LastFm database, similar in structure to the original LastFm database, containing only clean the tags for each track.'
    epilog = 'Example: python lastfm_cleaning.py ~/lastfm/lastfm_tags.db ~/lastfm/lastfm_tags_clean.db'
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("input", help=".db filename or path, or .csv folder path")
    parser.add_argument("output", help=".db filename or path")
    parser.add_argument('--supp-txt-path', help="set supplementary .txt folder path")
    parser.add_argument('--val', type=float, help="discard tags with val less than or equal to specified threshold")
    parser.add_argument('--is-clean', action='store_true', help='skip preliminary tag cleaning (that is, if the input database is already clean)')
    parser.add_argument('--matrix', help='if using --experimental, specify the path to the saved matrix (either the .npz path or the .nfo path will do, assuming the two files are in the same folder)')
    parser.add_argument('--exp-threshold', type=float, default=0.8, help='if using --experimental, set the correlation threshold to trigger the specified experimental function')
    parser.add_argument('--exp', type=int, choices={1, 2, 3}, help='perform advanced operations on the database, using one of the following functions: \
                                                                             1. if correlation between tag-A and tag-B is higher than a threshold, add tag-B \
                                                                             to all tracks that have tag-A but not tag-B; \
                                                                             2. if correlation between tag-A and tag-B is higher than a threshold, remove tag-A \
                                                                             from tracks which have both tag-A and tag-B; \
                                                                             3. if correlation between tag-A and tag-B is higher than a threshold, remove tag-B \
                                                                             from tracks which have both tag-A and tag-B')
    
    args = parser.parse_args()

    assert os.path.splitext(args.output)[1] == '.db', 'output filename must have .db extension'

    # check if output file already exists
    if os.path.isfile(args.output):
       raise FileNotFoundError("file " + args.output + " already exists!")

    def clean_database():
        # check if different .txt path has been provided
        if args.supp_txt_path:
            lastfm_utils.set_txt_path(args.supp_txt_path)

        # check if user provided a .csv folder or a .db file (if using .csv, load .csv into LastFm2Pandas; otherwise, load .db into LastFm)
        if os.path.isdir(args.input):
            try:
                tags = pd.read_csv(os.path.join(args.input, 'lastfm_tags.csv'))
            except FileNotFoundError:
                raise FileNotFoundError('Please make sure {} contains a file "lastfm_tags.csv".'.format(args.input))
            try:
                tids = pd.read_csv(os.path.join(args.input, 'lastfm_tids.csv'))
            except FileNotFoundError:
                raise FileNotFoundError('Please make sure {} contains a file "lastfm_tids.csv".'.format(args.input))
            try:
                tid_tag = pd.read_csv(os.path.join(args.input, 'lastfm_tid_tag.csv'))
            except FileNotFoundError:
                raise FileNotFoundError('Please make sure {} contains a file "lastfm_tid_tag.csv".'.format(args.input))
            lastfm = LastFm2Pandas.load_from(tags=tags, tids=tids, tid_tag=tid_tag)
        else:
            lastfm = LastFm(args.input)

        df = lastfm_utils.generate_final_df(lastfm)
        df.reset_index(drop=True, inplace=True) # sanity check
        df.index += 1

        assert all(df.columns == ['tag', 'merge_tags']) # sanity check

        # generate tables which will go into output database
        tags = df['tag'].str.lower()
        print('Matching all tags to the "clean" few ones...', end=' ', flush=True)
        tag_tag = create_tag_tag_table(lastfm, df)
        print('done')
        print('Matching all tids to tags...', end=' ', flush=True)
        tid_tag = create_tid_tag_table(lastfm, tag_tag, args.val)
        print('done')
        print('Purging tids...', end=' ', flush=True)
        tids = tid_tag['tid'].drop_duplicates()
        tids.index = tids.values
        tids = tids.map(lastfm.tid_num_to_tid).reindex(pd.RangeIndex(1, len(lastfm.get_tid_nums())+1))
        print('done')

        return lastfm.LastFm2Pandas.load_from(tags=tags, tids=tids, tid_tag=tid_tag) # wrap into LastFm2Pandas class

    if not args.exp:
        clean_fm = clean_database()
    else:
        from lastfm import Matrix
        
        if args.is_clean:
            clean_fm = LastFm2Pandas(args.input) # does not support .csv
        else:
            clean_fm = clean_database()
        
        # if matrix has already been computed, load it; otherwise, compute it from scratch
        if args.matrix:
            matrix = Matrix.load_from(args.matrix)
        else:
            matrix = Matrix(clean_fm, dim=3, tags=None)
        
        correlation = matrix.all_tag_is(args.exp_threshold)

        if args.exp == 1:
            clean_fm.tid_tag = experimental_clean_add(clean_fm.tid_tag, correlation) # the function experimental_clean_add() is not inplace; the two functions experimental_clean_remove() are
        elif args.exp == 2:
            experimental_clean_remove1(clean_fm.tid_tag, correlation)
        elif args.exp == 3:
            experimental_clean_remove2(clean_fm.tid_tag, correlation)
        
    # save into a new .db file
    print('Saving new tables as a .db file...', end=' ', flush=True)
    conn = sqlite3.connect(args.output)
    clean_fm.tags.to_sql('tags', conn, index=False)
    clean_fm.tids.to_sql('tids', conn, index=False)
    clean_fm.tid_tag.to_sql('tid_tag', conn, index=False)
    print('done')
    conn.close()
