'''Contains tools for merging and cleaning the lastfm_tags.db file


Notes
-----
The file aims to produce a dataframe that can be used to map the 
useful, clean, and meaningful tags onto the unprocessed tags in the tidtag 
dataset of the lastfm.db for training a neural network in later modules.


This file can be divided into three parts:
    
1. Convert the lastfm_tags.db into pd.DataFrame and use the generated 
dataframe to produce a popularity dataframe with columns:
    - ranking:
        The popularity ranking of the tag.
    - lastfm_ID:
        The ID of the tag in lastfm_tags.db.
    - tags:
        The tags.
    - count:
        The number of occurence of each tag.
    
2. Contains tools to produce a new dataframe with columns:
    - tag:
        The tags that will be used in training a neural network in later modules.
    - merge_tags:
        The tags from the lastfm_tags.db that will be merged into the 
        corresponding tag.
    Tools include adding new tags, combining existing tags, and remove tags, 
    merging similar dataframes and checking overlappings within the merge_tags 
    column and between the tag and merge_tags column.
    
3. Combine 1 and 2, and txt files for manually selecting useful tags and 
merge_tags.


Procedure
---------

The final dataset will contain two categories of tags stacked vertically. The
categories are:
    1. Genre tags:
        A threshold is set so that the tags from the lastfm_tags.db with 
        occurrences greater than or equal to the threshold will be kept. The 
        tags above the threshold can be returned by the 
        generate_genre_txt() function as a text file. A manual
        selection can be done by following the instruction provided printed
        after the generate_genre_txt() is run. Alternatively, you 
        may download the non_genre_list_filtered.txt list directly on Github 
        and save them in the txt_path directory.
        After the selection, by adjusting the parameters in the 
        generate_final_df() and run it, a clean dataset consisting of genre
        and vocal tags are generated.
        If after inspecting the generated dataframe you wish to add, 
        combine, drop tags or merge datasets, you may use the 
        add_tags(), combine_tags(), remove(), and merge_df() to manipulate
        the final dataset.
        
    2. Vocal tags:
        In this script, the default tags in this category are 
        ['female', 'instrumental', 'male', 'rap'].
        Firstly, generate_vocal_txt() will find a list of tags that are closely
        related to each tag and ouput as txt. After this, similar to the genre
        tags, a manual selection can be done by following the instruction
        provided printed after the generation_vocal_txt() function is run. 
        Alternatively, you may download the four txt files on Github and save 
        them in the txt_path directory.
        After the selection, by adjusting the parameters in the 
        generate_final_df() and run it, a clean dataset consisting of genre
        and vocal tags is generated.
        
        
IMPORTANT
----------
If you want to save a csv at any stage and what to continue the process later,
if there are columns consisting of lists, you will need to use eval() to 
convert the elements in the columns from str to list.
        

Functions
---------
- set_txt_path
    Set the output path to any csv files that will be saved, and the path of
    all the supplementary txt files.

- generate_vocal_txt
    Generate a txt file with a list of tags for each of the vocal tag filtered
    by percentile(). You may use this function to generate txt files for 
    vocal tags if you do not want to use the corresponding txt files on Github.

- generate_genre_txt
    Generate a txt file with a list of tags above the threshold that can be used
    to manually filter out non-genre tags. You may use this function to 
    generate a txt file for genre tagsif you do not want to use the 
    corresponding txt file on Github.

- clean_1
    Remove all non alphabet and number characters of the input string.

- clean_2
    Remove all non alphabet and number characters except '&' of the input 
    string, then replace '&' with 'n'.

- clean_3
    Remove all non alphabet and number characters except '&' of the input 
    string, then replace '&' with 'and'.
     
- clean_4
    Replace ' and ' with 'n', then remove all non alphabet and number 
    characters of the input string.

- clean_5
    Replace any 'x0s' string with '19x0s'.

- clean_6
    Replace any 'x0s' string with '20x0s'.

- check_plural
    Remove the trailing character 's' of the input string if its trailing part
    only contains only one 's'.

- check_overlap
    Check if there are any repeated tags in the merge_tags column of the 
    generated dataframe.

- combine_tags
    Combine given tags which exist in the tag column.
    
- add_tags
    Add new tags to the chosen tag row and further combine tags if a common tag
    appears simultaneously in two or more rows followed by combining the 
    overlapping tags in the merge_tags column.

- remove
    Remove a given tag from the dataset.

- merge_df
    Merge two or more output dataframes and ensure no overlappings between the
    tag column and the merge_tag column.

- search_genre
    Extend a given dataframe (df_output) for list of tags supplied by a list 
    (search_tags_list) by searching through the tags in the third dataframe 
    (df_input) using the given cleaning method. For more details, see function 
    description.

- percentile
    Return a dataframe with subset of tags (descending order) of the input 
    dataframe which accounts for a certain percentage of the total count of 
    all the tags.

- generate_vocal_df
    Return a dataframe based on the manually-filtered txt files provided for
    each of the vocal tags.

- generate_genre_df
    Combine all the genre related tools and various cleaning methods to generate a 
    clean dataframe of genre with no overlappings between tag and merge_tags 
    column and within the merge_tags column.

- generate_final_df
    Combine all the tools and generate the final dataframe consisting of 
    merging tags for each genre tag and vocal tag respectively.
'''

import numpy as np
import os
import pandas as pd
import re
import sqlite3
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.realpath(__file__))))

import query_lastfm as db

txt_path = '/srv/data/urop'
    
def set_txt_path(new_path):
    '''Set new_path as default path for opening all the supplementary txt files, and the output path
    of any csvs files.
    
    Parameters
    ----------
    new_path:
        path of the form /xxx/xx/xx .'''
        
    global txt_path
    txt_path = os.path.normpath(new_path)

def generate_vocal_txt(df: pd.DataFrame, tag_list = ['female', 'instrumental', 'male', 'rap'], percentage_list=[90, 90, 90, 80]):
    '''Generate a txt file with a list of tags for each of the vocal tag 
    filtered by percentile() that can be used to manually select merging tags
    for each tag in tag_list.
    
    Parameters
    ----------
    df: pd.DataFrame
        The popularity dataframe.
        
    tag_list: list
        The list of vocal tags that will be considered.
        
    percentage_list: list
        The percentage considered for each tag in the tag_list. The tags that
        is within the percentage will be output in the corresponding txt file 
        for each tag.
        
    Outputs
    -------
        Consists of a txt file for each of the tags in the tag_list. Each file consists of 
        all the tags filtered based on percentage_list.
        Please see the note printed after the function is run for instructions
        on how to work with the produced txt file.
    '''
    
    def generate_txt(df, tag, perc):
        df_thr = df[df['tag'].str.findall(r'\b'+tag, re.IGNORECASE).str.len()>0]
        df_thr = percentile(df_thr, perc=perc).tag.tolist()
        with open(os.path.join(txt_path, tag+'_list.txt'), 'w', encoding='utf8') as f:
            for item in df_thr:
                f.write("%s\n" % item)
    
    if len(tag_list) != len(percentage_list):
        raise ValueError('length of tag_list is unequal to length of percentage_list')
    
    for idx in range(len(tag_list)):
        generate_txt(df, tag_list[idx], percentage_list[idx])

    message = """Please deselect tags from generated txts by putting a "-" sign at\
                    the front of the term, or other symbol by adjusting the indicator\
                    input variable in the generate_vocal_df() function.\n \
                    e.g. If you want to deselect "female", replace "female" with "-female".\n \
                    Please rename the output files by adding a suffix "_filtered" to the filename and save in txt_path."""
    print(message)

def generate_genre_txt(df: pd.DataFrame, threshold: int = 20000):
    '''Generate a txt file with a list of tags above the threshold that can be 
    used to manually filter out non-genre tags.
    
    Parameters
    ----------
    threshold: int
        Tags with count greater than or equal to the threshold will be stored 
        as a txt file.
        
    csv_from_db: bool
        If True, the lastfm_tags.db will be converted to csv in order to 
        produce the popularity dataframe.
        
    Outputs
    -------
    txt file:
        Consists of all the tags above or equal to the threshold.
        Please see the note printed after the function is run for instructions
        on how to work with the produced txt file.
    '''
    
    tag_list = df['tag'][df['count'] >= threshold].tolist()

    with open(os.path.join(txt_path, 'non_genre_list.txt'), 'w', encoding='utf8') as f:
        for tag in tag_list:
            f.write("%s\n" % tag)
            
    message = """Please deselect tags from generated txts by putting a "-" sign at\
                    the front of the term, or other symbol by adjusting the indicator\
                    input variable in the generate_genre_df() function.\n \
                    e.g. If you want to deselect "rock", replace "rock" with "-rock".\n \
                    Please rename the output files as "non_genre_list_filtered.txt" and save in txt_path."""
    print(message)

def clean_1(tag):
    '''Remove all non alphabet and number characters of the input string.
    
    Parameters
    ----------
    tag: str
        A input string.
        
    Returns
    -------
    tag_sub: str
        The manipulated input tag.
    '''
    
    tag_sub = re.sub(r'[^A-Za-z0-9]', '', tag)
    return tag_sub

def clean_2(tag):
    '''Remove all non alphabet and number characters except '&' of the input string, then replace '&' with 'n'.
    
    Parameters
    ----------
    tag: str
        A input string.
        
    Returns
    -------
    tag_sub: str
        The manipulated input tag.
    '''
    
    tag_sub = re.sub(r'[^A-Za-z0-9&]', '', tag)
    tag_sub = re.sub('&', 'n', tag_sub)
    return tag_sub

def clean_3(tag):
    '''Remove all non alphabet and number characters except '&' of the input string, then replace '&' with 'and'.
    
    Parameters
    ----------
    tag: str
        A input string.
        
    Returns
    -------
    tag_sub: str
        The manipulated input tag.  
    '''
    
    tag_sub = re.sub(r'[^A-Za-z0-9&]', '', tag)
    tag_sub = re.sub('&', 'and', tag_sub)
    return tag_sub

def clean_4(tag):
    '''Replace ' and ' with 'n', then remove all non alphabet and number characters of the input string.
    
    Parameters
    ----------
    tag: str
        A input string.
        
    Returns
    -------
    tag_sub: str
        The manipulated input tag.
    '''
    
    tag_sub = re.sub(' and ', 'n', tag)
    tag_sub = re.sub(r'[^A-Za-z0-9]', '', tag_sub)
    return tag_sub

def clean_5(tag):
    '''Replace any 'x0s' string with '19x0s'.
    
    Parameters
    ----------
    tag: str
        A input string.
        
    Returns
    -------
    tag_sub: str
        The manipulated input tag.
    '''
    
    tag_sub = re.sub(r'[^A-Za-z0-9]', '', tag)
    tag_sub = re.sub(r'\b\d0s', lambda x:'19'+x.group(), tag_sub)
    return tag_sub

def clean_6(tag):
    '''Replace any 'x0s' string with '20x0s'.
    
    Parameters
    ----------
    tag: str
        A input string.
        
    Returns
    -------
    tag_sub: str
        The manipulated input tag.
    '''
    
    tag_sub = re.sub(r'[^A-Za-z0-9]', '', tag)
    tag_sub = re.sub(r'\b\d0s', lambda x:'20'+x.group(), tag_sub)
    return tag_sub

def check_plural(x):
    '''Remove the trailing character 's' of the input string if its trailing part only contains only one 's'.
    
    Parameters
    ----------
    x: str
        Input string.
    
    Returns
    -------
    str
        The input string with an 's' removed if there is only one 's' at the 
        end of the string.
    '''
    
    if len(x)<2:
        return x
    
    elif (x[-1]=='s') & (x[-2]!='s'):
        return x[:-1]
    
    else:
        return x

def check_overlap(df_input):
    '''Check if there are any repeated tags in the merge_tags column of a dataframe.
    
    Parameters
    ----------
    df_input: pd.DataFrame
        A dataframe with columns: 'tag', 'merge_tags'. The tags contained in 
        the 'tag' and 'merge_tags' column should be mutually exclusive.
        
        
    Returns
    -------
    df: pd.DataFrame
        The manipulated input dataframe so that there are no overlappings
        between the lists in the merge_tags columns.
    '''

    assert all([True if col in df_input.columns else False for col in ['tag', 'merge_tags']])
    df = df_input.copy()
    
    #indicator
    ind = 1
    # repeat the process until indicator becomes zero
    while ind==1:
    
        overlapped_idx_list = []
        #indicator, if no overlap occurs, ind remains 0
        ind = 0
    
        for idx, l in enumerate(df['merge_tags']):
            for idx2, l2 in enumerate(df['merge_tags'].iloc[:idx]):
                #check overlap
                if  set(l2).intersection(set(l)) != set():

                    #merge tags to upper list
                    df['merge_tags'].iloc[idx2] = list(set(l2+l))+[df['tag'].iloc[idx]]
                    #clear tags in lower list so no overlap because of this list in the future
                    df['merge_tags'].iloc[idx] = []

                    overlapped_idx_list.append(df.iloc[idx].name)
                    #change indicator to one if an overlap occurs
                    ind = 1
                    print(ind)
                    break
    
        #drop rows
        df = df.drop(overlapped_idx_list)
        
    return df

def combine_tags(df_input, list_of_tags, merge_idx=None):
    '''Combine given tags which exist in the tag column.
    
    Parameters
    ----------
    df_input: pd.DataFrame
        A dataframe with columns: 'tag', 'merge_tags'. The 'tag' column of 
        the dataframe should contain the tags that will be combined. All the 
        tags in the 'merge_tags' and 'tag' columns altogether should be unique. 
        
    list_of_tags: list
        A list of tags that will be combined in df_input. NOTE that all the 
        tags in the list must exist in the 'tag' column.
        
    merge_idx: 
        The index of the row which the resulting set of tags
        will be merged into. If None, the resulting set of tags will be merged
        to the row with the greatest popularity.
        
    Returns
    -------
    df: pd.DataFrame
        The dataframe produced after combining the tags. All the 
        tags in the 'merge_tags' and 'tag' columns altogether are unique, 
        provided that the df_input satisfies this condition.
    '''

    assert all([True if col in df_input.columns else False for col in ['tag', 'merge_tags']])
    assert all([True if tag in df_input.tag.tolist() else False for tag in list_of_tags])
    
    df = df_input.copy()
    
    list_of_tags = [item  if type(item)==str else str(item) for item in list_of_tags]
    
    #find the index of tag that all other tags will be merged into
    rows = df[df.tag.isin(list_of_tags)]
    
    if merge_idx:
        merge_index = merge_idx
    else:
        # merge into the row with greatest popularity
        merge_index = np.min(rows.index.tolist())

    
    for idx in range(rows.shape[0]):
        #merge all rows whose row number is not the merge_index
        if rows.iloc[idx,:].name != merge_index:
            row = rows.iloc[idx,:]

            df.loc[merge_index, 'merge_tags'] = list(set(row.merge_tags+[row.tag]+df.loc[merge_index, 'merge_tags']))
            df = df.drop(row.name)
        
    return df

def add_tags(df_input, list_of_tags, target_tag, target_merge_index=True):
    '''Add new tags to the chosen tag row and further combine tags if a common tag
    appears simultaneously in two or more rows followed by combining the 
    overlapping tags in the merge_tags column.
    
    Parameters:
    -----------
    df_input: pd.DataFrame
        A dataframe with columns: 'tag', 'merge_tags'. One of the columns of 
        the dataframe should contain the tags that will be added. All the 
        tags in the 'merge_tags' and 'tag' columns altogether should be unique.
        
    list_of_tags: list
        A list of tags that will be combined. Note that if the list contains 
        the target tag, the target tag will be removed automatically in the 
        function.
        
    target_tag: str/float
        The tag in the 'tag' column of the df_input which all the tags in 
        the list_of_tags will be merged into.
        
    target_merge_index: bool
        If True, when combining tags using the combine_tags() function is 
        necessary, the resulting list of merging tags will be merged into
        the target_tag row. If False, the resulting list of merging tags will
        be merged into the row with greatest popularity.
    
    Returns
    -------
    df: pd.DataFrame
        The dataframe produced after adding the tags. In this function, tags 
        in the list_of_tags which do not exist in the 'tag' column will be
        directly merged into the target_tag row first. The remaining tags in 
        list will follow the same procedure in the combine_tags() function to 
        ensure no overlappings between the 'tag' column and the 'merged_tags'
        column. Finally, check_overlap() functon will be used to deal with 
        the overlapping issues within the 'merged_tags' columns. Hence 
        all the tags in the 'merge_tags' and 'tag' columns altogether are 
        unique, provided that the df_input satisfies this condition.
    '''

    assert all([True if col in df_input.columns else False for col in ['tag', 'merge_tags']])
    
    df = df_input.copy()
    
    list_of_tags = [item  if type(item)==str else str(item) for item in list_of_tags]
        
    if type(target_tag) != str:
        target_tag = str(target_tag)
    
    if target_tag not in df.tag.tolist():
        #find index of the row the tag belongs to
        _ = [idx  for idx, row in enumerate(df.merge_tags) if target_tag in row]
        
        #error handling
        if len(_)>0:
            _ = _[0]
        else:
            raise ValueError('target tag does not exist')
            
        target_tag = df_input.iloc[_].tag
        
    if target_tag in list_of_tags:
        list_of_tags.remove(target_tag)
    
    #find tags that exist in both df['tag'] and list_of_tags
    common_tags = list(set(df.tag.tolist()).intersection(set(list_of_tags)))
    common_tags.append(target_tag)
    
    #compliment of common_tags in list_of_tags
    remain_tags = [tag for tag in list_of_tags if tag not in common_tags]

    # the merge_tags list of the row that other tags will be merged into
    target_row = df[df.tag==target_tag]
    target_list = target_row.iloc[0].merge_tags
    
    #merge remain_tags into the row first
    df.loc[df.tag==target_tag, 'merge_tags'] = [list(set(target_list+remain_tags))]
    
    # combine common_tags to the row
    if target_merge_index:
        merge_idx = target_row.index[0]
        df = combine_tags(df, common_tags, merge_idx)
    else:
        df = combine_tags(df, common_tags)
    
    df = check_overlap(df)
    
    return df

def remove(df_input, tag):
    '''Remove a given tag from the dataset.
    
    Parameters
    ----------
    df_input: pd.DataFrame
        A dataframe with columns: 'tag', 'merge_tags'. One of the columns of 
        the dataframe should contain the tag that will be added.
        
    tag: str/float
        The tag that you wish to be removed in the dataframe.
        
    Returns
    -------
        The dataframe with the tag removed.
    '''
    
    assert all([True if col in df_input.columns else False for col in ['tag', 'merge_tags']])
    
    if type(tag) != str:
            tag = str(tag)
    
    df = df_input.copy()
    if tag in df.tag.tolist():
        df = df.drop(df[df.tag==tag].index[0])
    else:
        #find index of the row the tag belongs to
        _ = [idx  for idx, row in enumerate(df.merge_tags) if tag in row][0]
        col = list(df.columns).index('merge_tags')
        new_list = df.iloc[_,col]
        df.iloc[_,col]= [item for item in new_list if item != tag]
    
    return df

def merge_df(list_of_df):
    '''Merge two or more output dataframes and ensure no overlappings between the
    tag column and the merge_tag column.
    
    Parameters
    ----------
    list_of_df: list of pd.DataFrame
        The dataframes that will be merged with columns: tag', 'merge_tags'. 
        All the tags in the 'merge_tags' and 'tag' columns altogether in each 
        dataframe should be unique respectively.
        
    Returns
    -------
    df_merge: pd.DataFrame
        The resulting dataframe. All the tags in the 'merge_tags' and 'tag' 
        columns altogether are unique, provided that all the dataframes
        provided in the list satisfy this condition. The resulting dataframe
        is produced by pandas outer merge followed by combine_tags() function 
        for tags that only exist in the 'tag' column of some of the dataframes, 
        but in the mean time exist in the 'merge_tags' column of one or more of 
        the remaining dataframes.
    
    '''

    # combine elements in two columns for later use
    def row_op(row):
        col1, col2 = row.index[-2], row.index[-1]
        list1, list2 = row[col1], row[col2]
        if type(list1) != list:
            list1 = []

        if type(list2) != list:
            list2 = []

        return list(set(list1+list2))
    
    for df in list_of_df:
        assert all([True if col in df.columns else False for col in ['tag', 'merge_tags']])
    
    df = list_of_df[0].copy()
    
    for df2 in list_of_df[1:]:
        # outer merge and use row_op to combine the two list of merge_tags
        df_merge = df.merge(df2, how='outer', on='tag')
        df_merge['merge_tags'] = df_merge.apply(row_op, axis=1)
        df_merge = df_merge[['tag','merge_tags']]
        
        # search for tags that only exist in one of df and df2
        new_tags = []
        new_tags.append(list(set(df_merge.tag).difference(set(df.tag))))
        new_tags.append(list(set(df_merge.tag).difference(set(df2.tag))))
        new_tags = [i for sublist in new_tags for i in sublist]
        
        # make sure new standalone tags do not exist in any of the merge_tags lists (there is no need 
        # to check for old tags because we have already ensured that old common tags and all tags contained in merge_tags column is 
        # mutually exclusive)
        for tag in new_tags:
            for idx, l in enumerate(df_merge.merge_tags):
                if tag in l:
                    #g et the corresponding tag in the tag column
                    tag2 = df_merge.iloc[idx, :].tag
                    df_merge = combine_tags(df_merge, [tag, tag2], merge_idx=idx)
                    
                    break

    df_merge = check_overlap(df_merge)
    
    return df_merge

def percentile(df, perc=90):
    '''Return a dataframe with subset of tags (descending order) of the input 
    dataframe which accounts for a certain percentage of the total count of 
    all the tags.
    
    Parameters
    ----------
    df: pd.DataFrame
        A dataframe with columns: 'tag', 'count'.
        
    perc: float
        The percentage of the total count of all the tags that will be 
        considered.
        
    Returns
    -------
    df: pd.DataFrame
        The input df that is cut based on the perc parameter.
    
    '''

    assert all([True if col in df.columns else False for col in ['tag', 'count']])
    tot_counts = df['count'].sum()
    threshold = perc * tot_counts /100
    counter = 0
    
    for i, row in enumerate(df['count']):
        if counter < threshold:
            counter += row
        else:
            return df.iloc[:i,]

def search_genre(df_input, df_output, search_method=clean_1, search_tags_list=None, 
                 sub_threshold=10, verbose=True, remove_plural=True):
    
    '''Extend a given dataframe (df_output) for list of tags supplied by a list 
    (search_tags_list) by searching through the tags in the third dataframe 
    (df_input) using the given cleaning method. For more details, see function 
    description.
    
    Parameters
    ----------
    df_input: pd.DataFrame
        The popularity dataframe or a dataframe with similar structure, i.e. 
        with columns 'tag', 'count'. This dataframe provides a list of tags
        as a pool where another tag can find corresponding matching tags from.
        
    df_output: pd.DataFrame
        A dataframe with columns: 'tag', 'merge_tags'. This is the dataframe 
        which you want to search for new tags from df_input and add to
        appropriate rows of the 'merge_tags' column. All the tags in the 
        'merge_tags' and 'tag' columns altogether should be unique. If you wish
        to generate a new dataframe from scratch, df_output should contain two
        columns: 
            1. 'tag' -- contains all the desired tags that a search will
            be run on to find matching tags. (It may contain tags which a 
            search will not be run on. You may specify what tags you wish to 
            be run on by specifying the search_tags_list parameter.) 
            i.e., search_tags_list must be a subset of the list of tags in 
            the 'tag' column.
            
            2. 'merge_tags'  -- fill all the entry with empty list [ ].  
            
            For example:
        
            search_tags_list = df_input[df_input['count']>=threshold]
        
            df_output = pd.DataFrame({'tag':search_tags_list['tag'].tolist(), 'merge_tags':[[]]*len(search_tags_list)})
        
    search_method: func
        A function that takes a string and transform that to a new string.
        clean_1, clean_2 etc. are some examples.
        
    threshold: int
        Searches will be run on tags with count above or equal to the 
        threshold and merge into the df_output dataframe. If threshold=None, 
        searches will be run on all the tags in the df_output. Note that 
        threshold does not have any effect if search_tags_list is not None.
    
    search_tags_list: pd.DataFrame
        A reduced version of the popularity dataframe or a dataframe with 
        columns: 'tag', 'count'. The 'tag' column consists of all the tags
        that a search will be run on. None that f search_tags_list is not None, threshold 
        will have no effect.
        
    sub_threshold: int
        Only the tags with count greater than or equal to sub_threshold in the
        df_input dataframe will be in the search pool. If sub_threshold=None, 
        sub_threshold is assumed to be zero.
        
    verbose: bool
        If True, print progress.
        
    remove_plural: bool
        If True, the check_plural functions are applied to all the tags 
        involved in the search.
        
    Returns
    -------
    df_output: pd.DataFrame
        The dataframe after tags merging and adding from the search pool.
        The dataframe is produced under the following procedures:
            
        1. A search is run on for each tag of a list of tags specified by the 
        parameters above. Each search works by converting all the tags based
        on the cleaning_method, and remove_plural into new set of tags. Note
        that each search ignores lower case or upper case. Each search is done 
        by using the pool of tags provided by df_input, restricted by sub_threshold.
        
        2. The algorithm then use the add_tags() functions to integrate the
        search list into df_output for each tag a search is run on.
        
        All the tags in the 'merge_tags' and 'tag' columns altogether are 
        unique, provided that the df_output provided as a parameter satisfies 
        this condition.
    '''

    assert all([True if col in df_output.columns else False for col in ['tag', 'merge_tags']])
    assert all([True if col in df_input.columns else False for col in ['tag', 'count']])

    df = df_input.copy()
    df['tag'] = df['tag'].astype('str')
    df['tag_sub'] = df['tag'].apply(search_method)

    if remove_plural:
        df['tag_sub'] = df['tag_sub'].apply(check_plural)
    
    if search_tags_list is None:
        search_tags_list = df_output.tag.tolist()
    
    tot = len(search_tags_list)
    if sub_threshold is not None:
        bool1 = df['count']>=sub_threshold
    else:
        bool1 = df['count']>=0
    
    for num, tag in enumerate(search_tags_list):
        
        # remove non-word characters
        tag_sub = search_method(tag)
        if remove_plural:
            tag_sub = check_plural(tag_sub)
            
        # query
        search = r'^'+tag_sub+r'$'
            
        # get list of tags that will be merged (tag such that its 'pure' form matches with tag_sub)
        bool2 = df['tag_sub'].str.findall(search, re.IGNORECASE).str.len()>0
            
        merge_tags = df[(bool1 & bool2)]['tag'].tolist()
        merge_tags = [item for item in merge_tags if item != tag] 
            
        df_output = add_tags(df_output, merge_tags, tag, False)
            
        if verbose:
            if (num+1)%10==0:
                print('processed {} out of {} tags'.format(num+1, tot))
                
    return df_output

def generate_genre_df(popularity: pd.DataFrame, threshold: int = 2000, sub_threshold: int = 200, verbose=True, drop_list_filename='non_genre_list_filtered.txt', indicator='-'):
    
    '''Combine all genre related tools and various cleaning methods to generate a 
    clean dataframe of genre with no overlappings between tag and merge_tags 
    column and within the merge_tags column. For more details, see
    documentation on search_genre().
    
    Parameters
    ----------
    csv_from_db: bool
        If True, the lastfm_tags.db will be converted to csv in order to 
        produce the popularity dataframe.
        
    threshold: int
        Searches will be run on tags with count above or equal to the 
        threshold.
        
    sub_threshold: int
        Only the tags with count greater than or equal to sub_threshold in the
        will be in the search pool. If sub_threshold=None, sub_threshold is assumed to 
        be zero.
        
    verbose: bool
        If True, print progress.
        
    drop_list_filename: str
        The filename of the txt file that stores the information of whether
        a tag is considered as a genre tag. You may download the document
        on Github, or produce one using generate_genre_txt(). 
        For more details please refer to the function documentation of 
        generate_genre_txt(). NOTE that the file should be saved
        under the directory specified by the variable txt_path.
        
    indicator: str (one character)
        A special symbol used at the front of some tags in 
        non_genre_list_filtered.txt to denote non-genre tags. Default '-'.
        
    Returns
    -------
    df_filter: pd.DataFrame
        The dataframe with columns: tag', 'merge_tags'. The tag column consists
        of tags with count greater than or equal to threshold. 
        This is produced based on the popularity dataframe, by applying 
        the search_genre() function using clean_1 up to clean_6. All the tags 
        in the 'merge_tags' and 'tag' columns altogether are unique.
    '''
    
    def search_matching_items(l, regex):
        output_list = []
        for item in l:
            item = re.search(regex, item)
            if item is not None:
                output_list.append(item.group(0))
        return output_list
    
    print('Genre progress 1/7  --dropping tags')
    txt_path = os.path.join(txt_path, drop_list_filename)
    drop_list = []
    with open(txt_path, 'r', encoding='utf8') as f:
        for item in f:
            if ((item[0]==indicator) and (not item.isspace()) and (item!='')):
                drop_list.append(item.rstrip('\n').lstrip(indicator))
                
    df = popularity.copy()
    
    # drop tags
    df.tag = df.tag.astype(str)
    df = df[-df.tag.isin(drop_list)]
    
    search_tags_list = df['tag'][df['count'] >= threshold].tolist()
    
    print('Genre-Step 2/7  --cleaning_1')
    # generate empty dataframe structure
    df_output = pd.DataFrame({'tag':search_tags_list, 'merge_tags':[[]]*len(search_tags_list)})
    
    df_filter = search_genre(df, df_output, search_method=clean_1, search_tags_list=None,
                             sub_threshold=sub_threshold, verbose=verbose)
    
    print('Genre-Step 3/7  --cleaning_2')
    
    search_tags_list = search_matching_items(search_tags_list, r'.*&.*')

    df_filter = search_genre(df, df_filter, search_method=clean_2, 
                             sub_threshold=sub_threshold, search_tags_list=search_tags_list, verbose=verbose)
    print('Genre-Step 4/7  --cleaning_3')
    df_filter = search_genre(df, df_filter, search_method=clean_3,
                             sub_threshold=sub_threshold, search_tags_list=search_tags_list, verbose=verbose)
    
    print('Genre-Step 5/7  --cleaning_4')
    search_tags_list = df['tag'][df['count'] >= threshold].tolist()
    search_tags_list = search_matching_items(search_tags_list, r'.* and .*')
    df_filter = search_genre(df, df_filter, search_method=clean_4,
                             sub_threshold=sub_threshold, search_tags_list=search_tags_list, verbose=verbose)
    
    print('Genre-Step 6/7  --cleaning_5')
    search_tags_list = df['tag'][df['count'] >= threshold].tolist()
    search_tags_list = search_matching_items(search_tags_list, r'\b\d0s')
    df_filter =  search_genre(df, df_filter, search_method=clean_5,
                              sub_threshold=sub_threshold, search_tags_list=search_tags_list, 
                              remove_plural=False, verbose=verbose)
    
    print('Genre-Step 7/7  --cleaning_6')
    df_filter =  search_genre(df, df_filter, search_method=clean_6,
                              sub_threshold=sub_threshold, search_tags_list=search_tags_list, 
                              remove_plural=False, verbose=verbose)
    
    print('Genre--Done')
    
    return df_filter

def generate_vocal_df(indicator='-', 
                      tag_list = ['female', 'instrumental', 'male', 'rap']):
    '''Return a dataframe based on the manually-filtered txt files provided for
    each of the vocal tags.
    
    Parameters
    ----------
    indicator: str (One character)
        A special symbol used at the front of some tags in 
        the txt file for each tag in the tag_list to denote tags that will be 
        drop. Default '-'.
    
    tag_list: list
        A list of vocal tags that is considered. (Shold be consistent with
        the txt files stored.)
        
    Returns
    -------
    df_filter: pd.DataFrame
        The output dataframe with columns: 'tag', 'merge_tags', the same format
        as the one generated by generate_genre_df() for genre. All the tags 
        in the 'merge_tags' and 'tag' columns altogether are unique.
    '''
    
    def load_txt(filename):
        path = os.path.join(txt_path, filename)
    
        tags = []
        with open(path, 'r', encoding='utf8') as f:
            for item in f:
                if ((item[0]!=indicator) and (not item.isspace()) and (item!='')):
                    tags.append(item.rstrip('\n'))
        return tags
    
    print('Vocal-Step 1/1')
    merge_tags_list=[]
    
    for filename in tag_list:
        tags= load_txt(filename+'_list_filtered.txt')

        
        if filename in tags:
            tags.remove(filename)
            
        merge_tags_list.append(tags)

    df_filter =  pd.DataFrame({'tag':tag_list, 'merge_tags':merge_tags_list})
    
    print('Vocal--Done')
    return df_filter

def generate_final_df(lastfm=None, from_csv_path='/srv/data/urop/', from_csv_path_split=['lastfm_tags.csv', 'lastfm_tids.csv', 'lastfm_tid_tag.csv'], 
                       verbose=True, threshold=2000, sub_threshold=100,
                       pre_drop_list_filename='non_genre_list_filtered.txt',
                       combine_list=[['rhythm and blues', 'rnb'], ['funky', 'funk']], 
                       drop_list=['2000', '00', '90', '80', '70', '60'],
                       genre_indicator='-', vocal_indicator='-',
                       vocal_tag_list=['female', 'instrumental', 'male', 'rap'],
                       add_list=None, add_target=None, add_target_merge_index=True):
    '''Combine all the tools and generate the final dataframe consisting of 
    merging tags for each genre tag and vocal tag respectively.
    
    Parameters
    ----------
    lastfm: db.LastFm, db.LastFm2Pandas
        Instance of the database class to produce the popularity dataframe.

    from_csv_path: str
        If an instance of the database class is not available, create a new instance
        from scratch given a set of csv files located in from_csv_path. 
    
    from_csv_path_split: list
        If an instance of the database class is not available, create a new instance
        from scratch given a set of csv files located in from_csv_path, with filename
        contained in from_csv_path_split (expecting a list of 3 filenames).
        
    threshold: int
        Only the tags with count greater than or equal to threshold in the popularity df
        will be searched through.
        
    sub_threshold: int
        Only the tags with count greater than or equal to sub_threshold in the popularity df
        will be in the search pool for genre tag. 
        If sub_threshold=None, sub_threshold is assumed to be zero.
        
    verbose: bool
        If True, print progress.
        
    pre_drop_list_filename: str
        The filename of the txt file that stores the information of whether a tag is considered as a genre tag. 
        You may download the document on GitHub, or produce one using generate_genre_txt(). 
        For more details please refer to the function documentation of 
        generate_genre_txt(). The file should be saved
        under the directory specified by txt_path.
        
    combine_list: list of list
        Each sublist contains a set of tags that will be combined by the 
        combine_tags() function. See documentation on combine_tags() for more
        details.
        
    drop_list: list
        A list of tags that will be dropped from the output dataframe. See
        documentation on remove() for more details.
        
    genre_indicator: str (One character)
        A special symbol used at the front of some tags in 
        non_genre_list_filtered.txt to denote non-genre tags. Default '-'. For
        more details see documentation on generate_genre_txt() and
        generate_genre_df().
        
    vocal_indicator: str (One character)
        A special symbol used at the front of some tags in 
        the txt file for each tag in the tag_list to denote tags that will be 
        drop. Default '-'. For more details see documentaton on
        generate_vocal_txt() and generate_vocal_df(). Note that the txt files
        for the tags in tag_list should be saved under the same directory
        as the variable 'txt_path'.
        
    add_list: list of list
        Each sublist contains the tags that will be added to the output 
        dataframe. For more details, see documentation on add_tags(). If None,
        nothing will be added.
        
    add_target: list
        The target tag for each sublist in the add_list parameter. See 
        documentation on add_tags() for more details. If add_list is None, this
        parameter will have no effects.
        
    add_target_merge_index: bool or list of bool
        If True for a speicifc target_tag, when combining tags using the 
        combine_tags() function during the adding tags process is necessary, 
        the resulting list of merging tags will be merged into the target_tag 
        row provided by add_target variable. If False, the resulting list of 
        merging tags will be merged into the row with greatest popularity for 
        each target tag. If only one bool (True or False) is input, it is 
        assumed that the bool is same for all the target_tags. See 
        documentation on add_tags for more details. If add_list is None, this
        parameter will have no effects.
    
    Returns
    -------
    df_final: pd.DataFrame
        A dataframe consisting of two smaller dataframes vertically stacked.
        The two smaller dataframes are returned by generate_vocal_df() and 
        generate_genre_df() based on the parameters provided.
    '''

    if lastfm is not None:
        df = lastfm.popularity()
    else:
        assert len(from_csv_path_split) == 3
        lastfm = db.LastFm2Pandas.from_csv(from_csv_path, from_csv_path_split)
        df = lastfm.popularity()
    
    vocal = generate_vocal_df(indicator=vocal_indicator)
    genre = generate_genre_df(popularity=df, threshold=2000,
                              sub_threshold=sub_threshold, verbose=verbose,
                              drop_list_filename=pre_drop_list_filename,
                              indicator=genre_indicator)
    
    df_final = pd.concat([genre, vocal])
    
    for item in combine_list:
        df_final = combine_tags(df_final, item)
        
    for item in drop_list:
        df_final = remove(df_final, item)
    
    if add_list is not None:
        if len(add_list)!=len(add_target):
            raise ValueError('length of add_list is unequal to length of add_target.')
        
        for idx in range(len(add_list)):
            if len(add_target_merge_index)==1:
                df_final = add_tags(df_final, add_list[idx], add_target[idx],
                                    target_merge_index=add_target_merge_index)
            
            if len(add_list)>1 and len(add_target_merge_index) == len(add_list):
                df_final = add_tags(df_final, add_list[idx], add_target[idx],
                                    target_merge_index=add_target_merge_index[idx])
                
            if len(add_list)!=len(add_target_merge_index):
                raise ValueError('lenght of add_list is unequal to length of add_target_merge_index')
    
    df_final = df_final.reset_index(drop=True)
    
    print('Done --dataframe generated')
    return df_final