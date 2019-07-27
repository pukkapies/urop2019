import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../modules')))

import query_lastfm as db

# from aden import his module

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
    output['tag'] = output['tag'].apply(lambda t: db.tag_to_tag_num(t))
    output.columns = ['tag_num', "new_tag_num"]

    # append row of 0's at the top
    nul = pd.DataFrame(data={'tag_num': [0], 'new_tag_num': [0]})
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
    flat = flat.set_index('tag_num', verify_integrity=True).sort_index()
    
    # fetch the tag num's from the original database
    tag_tag = pd.Series(db.get_tag_nums())
    tag_tag.index = tag_tag.index + 1

    # define a new 'loc_except' function that locates an entry if it exists, and returns 0 otherwise
    flat['new_tag_num'].loc_except = locate_with_except(flat['new_tag_num'])

    # for each tag num, get the corresponding 'clean' tag num from the flattened dataframe (returns tag num = 0 if tag falls below the pop threshold)
    tag_tag = tag_tag.apply(lambda i: flat['new_tag_num'].loc_except(i))

    return tag_tag

def create_tid_tag_table(db: db.LastFm, df_tag_tag: pd.DataFrame):
    '''
    Create a dataframe with two columns: 'tid' contains all the tid's from the original tid_tag table,
    'tag' contains the new tag. Here all the 0's are dropped.
    '''

    # fetch the tids from the original database, and map the tags to their correspondent 'clean' tags
    col_1 = db.tid_tag['tid']
    col_2 = db.tid_tag['tag'].apply(lambda t: df_tag_tag['new_tag'].loc[t])

    # concatenate columns in a dataframe
    output = pd.concat([col_1, col_2], axis=1)

    # remove tags which fall below the popularity theshold
    output = output[output['tag'] != 0]
    return output

if __name__ == "__main__":

    # do an argvparse to get lastfm_db location and output location for the clean_lastfm_db

    # open sql connection as conn

    # create an instance of LastFm class (will be in the new query_lastfm_pd.py module)
    # lastfm = db.LastFm(path-to-db)
    
    # df = aden module .generate_df()
    df.reset_index(inplace=True)
    df.index = df.index + 1 # love pandas

    tags = df['tags'] # read the 'clean' tags table straight from Aden's dataframe
    tag_tag = create_tag_tag_table(lastfm, df)
    tid_tag = create_tid_tag_table(lastfm, tag_tag)

    # tags.to_sql('tags', conn)
    # tag_tag.to_sql('tag_tag', conn)
