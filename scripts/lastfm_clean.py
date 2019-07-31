''' Contains tool to generate a new lastfm_tags.db file containing only the "clean" merged tags


Notes
-----
This file can be run as a script. To do so, just type 'python clean_lastfm.py' in the terminal. The help 
page should contain all the options you might possibly need. 

The script relies on the query_lastfm module to perform queries on the original database, and on the
lastfm_tool module to obatain a dataframe with the tags to retain and the tags to merge.

The script will output a db file similar in structure to the original lastfm_tags.db. The query_lastfm
module will work on the new db file as well.


Functions
---------
- flatten
    Take the tag dataframe and flatten the tags of the merge-tags column into separate rows.

- flatten_to_tag_num
    Take the tag dataframe and flatten, then replace tag with tag nums from the original database, then add one row of 0's at the top of the output dataframe.

- create_tag_tag_table
    Create a dataframe linking the tag num's in the original lastfm database to the tag num's in the new "clean" database.

- create_tid_tag_table
    Create a dataframe containing all the "clean" tags for each tid.
'''

import argparse
import ast
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../modules')))

import query_lastfm as q_fm

def flatten(df: pd.DataFrame):
    ''' Produce a dataframe with the following columns: 'tag', 'new_tag_num'.

    Parameters
    ----------
    df : pd.DataFrame
        The new tags dataframe to flatten.

    Examples
    --------
    If the original df looks like...

        |    tags        |  merge-tags
    ---------------------------------------------
    1   |    rock        |  ['ROCK']
    2   |    pop         |  []
    3   |    hip hip     |  ['hiphop', 'hip-hop']

                                ...then flatten(df) returns:

        |    tag         |  tag_num
    ---------------------------------------------
    1   |   rock         |  1
    2   |   ROCK         |  1
    3   |   pop          |  2
    4   |   hip hop      |  3
    5   |   hiphop       |  3
    6   |   hip-hop      |  3
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

def flatten_to_tag_num(lf: q_fm.LastFm, df: pd.DataFrame):
    ''' Produce a dataframe with the following columns: 'tag_num', 'new_tag_num'. The tags in the tag column in flatten() are substituted by their original tag nums.
    
    Parameters
    ----------
    db : q_fm.LastFm, q_fm.LastFm2Pandas
        Any instance of the tags database.

    df : pd.DataFrame
        The new tags dataframe to flatten.
    '''

    output = flatten(df)
    output['tag'] = output['tag'].map(lf.tag_to_tag_num)
    output.columns = ['tag_num', 'new_tag_num']

    # append row of 0's at the top
    nul = pd.DataFrame(data={'tag_num': [0], 'new_tag_num': [0]})
    output.index += 1 # otherwise output.append(verify_integrity=True) returns an error
    output = output.append(nul, verify_integrity=True).sort_index()
    return output

def create_tag_tag_table(lf: q_fm.LastFm, df: pd.DataFrame):
    '''
    Produce a dataframe with the following columns: 'tag_num', 'new_tag_num'. This will contain all the tags from the original tags database, and 0 as a new tag if the old tag has been discarded.
    
    Parameters
    ----------
    db : q_fm.LastFm, q_fm.LastFm2Pandas
        Any instance of the tags database.

    df : pd.DataFrame
        The new tags dataframe.
    '''
    
    def locate_with_except(series):
        def fn(i):
            try:
                return series.loc[i]
            except KeyError:
                return 0
        return fn

    flat = flatten_to_tag_num(lf, df)
    flat['index'] = flat.index.to_series()
    flat = flat.set_index('tag_num', verify_integrity=True).sort_index()
    
    # fetch the tag num's from the original database
    tag_tag = pd.Series(lf.get_tag_nums())
    tag_tag.index += 1

    # define a new 'loc_except' function that locates an entry if it exists, and returns 0 otherwise
    loc_except = locate_with_except(flat['new_tag_num'])

    # for each tag num, get the corresponding 'clean' tag num from the flattened dataframe (returns tag num = 0 if tag falls below the pop threshold)
    tag_tag = tag_tag.map(loc_except)
    return tag_tag

def create_tid_tag_table(lf: q_fm.LastFm, tag_tag: pd.DataFrame, tid_tag_threshold: int = None):
    '''
    Produce a dataframe with the following columns: 'tid', 'tag'. 

    Parameters
    ----------
    db : q_fm.LastFm, q_fm.LastFm2Pandas
        Any instance of the tags database.

    tag_tag : pd.DataFrame
        The tag_tag dataframe produced with create_tag_tag_table().
    
    tid_tag_threshold : int
        The minimum value for val (see original tags database) to allow in the new tid_tag table.
    '''
    
    # fetch the tids from the original database, and map the tags to their correspondent 'clean' tags
    if tid_tag_threshold is not None:
        tid_tag = lf.fetch_all_tids_tags_threshold(tid_tag_threshold)
    else:
        tid_tag = lf.fetch_all_tids_tags()
    
    if not isinstance(tid_tag, pd.DataFrame): # type(tid_tag) varies depending on whether q_fm.LastFm or q_fm.LastFm2Pandas is being used
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

    import lastfm_tool as lf_tool

    description = "Script to generate a new LastFm database, similar in structure to the original LastFm database, containing only clean the tags for each track."
    epilog = "Example: python clean_lastfm.py ~/lastfm/lastfm_tags.db ~/lastfm/lastfm_tags_clean.db"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("input", help="input db filename or path, or csv folder path")
    parser.add_argument("output", help="output db filename or path")
    parser.add_argument('--val-thresh', dest='val', type=float, help="discard tags with val less than threshold")
    parser.add_argument('--supp-txt-path', help="path to supplementary txt folder")
    
    args = parser.parse_args()
    
    # if user provided a csv folder, load csv into LastFm2Pandas; otherwise, load db into LastFm
    if os.path.isdir(args.input):
        lastfm = q_fm.LastFm2Pandas.from_csv(args.input)
    else:
        lastfm = q_fm.LastFm(args.input)

    # check if output ends with db extension
    if args.output[-3:] != '.db':
        args.output += '.db'

    # check if output already exists
    if os.path.isfile(args.output):
       print("WARNING file " + args.output + " already exists!")
       sys.exit(0)
    
    # check if different txt path has been provided
    if args.supp_txt_path:
        lf_tool.set_output_path(args.supp_txt_path)
    
    df = lf_tool.generate_final_df(lastfm)

    assert all(df.columns == ['tag', 'merge_tags'])

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
