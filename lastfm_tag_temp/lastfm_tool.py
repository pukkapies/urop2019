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
    - counts:
        The number of occurence of each tag.
    
2. Contains tools to produce a new dataframe with columns:
    - tag:
        The tags that will be used in training a neural network in later modules.
    - merge_tags:
        The tags from the lastfm_tags.db that will be merged to the corresponding
        tag.
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
        generate_non_genre_droplist_txt() function as a text file. A manual
        selection can be done by following the instruction provided printed
        after the generate_non_genre_droplist_txt() is run. Alternatively, you 
        may download the non_genre_list_filtered.txt list directly on Github 
        and save them in the output_path directory.
        After the selection, by adjusting the parameters in the 
        generate_final_csv() and run it, a clean dataset consisting of genre
        and vocal tags are generated.
        If after inspecting the generated dataframe you wish to add, 
        combine, drop tags or merge datasets, you may use the 
        add_tags(), combine_tags(), remove_tag(), and merge_df() to manipulate
        the final dataset.
        
    2. Vocal tags:
        In this script, the default tags in this category are 
        ['female', 'instrumental', 'male', 'rap'].
        Firstly, generate_vocal_txt() will find a list of tags that are closely
        related to each tag and ouput as txt. After this, similar to the genre
        tags, a manual selection can be done by following the instruction
        provided printed after the generation_vocal_txt() function is run. 
        Alternatively, you may download the four txt files on Github and save 
        them in the output_path directory.
        After the selection, by adjusting the parameters in the 
        generate_final_csv() and run it, a clean dataset consisting of genre
        and vocal tags is generated.
        
        
IMPORTANT
----------
If you want to save a csv at any stage and what to continue the process later,
if there are columns consisting of lists, you will need to use eval() to 
convert the elements in the columns from str to list.
        

Functions
---------
- set_path
    Set path to the lastfm_tags.db.
    
- set_output_path
    Set the output path to any csv files that will be saved, and the path of
    all the supplementary txt files.
    
- db_to_csv
    Convert lastfm_tags.db into a pd.DataFrame.

- insert_index
    Insert tags and TID index to dataframes generated from lastfm_tags.db.

- popularity
    Generate the popularity dataset. For more details, see function 
    description.

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

- remove_tag
    Remove a given tag from the dataset.

- search_genre_exact
    Extend a given dataframe (df_output) for list of tags supplied by another
    dataframe (df_thr) by searching through the tags in the third dataframe 
    (df_input) using the given cleaning method. For more details, see function 
    description.
    
- merge_df
    Merge two or more output dataframes and ensure no overlappings between the
    tag column and the merge_tag column.

- generate_non_genre_droplist_txt
    Generate a txt file with a list of tags above the threshold that can be used
    to manually filter out non-genre tags.

- generate_genre_df
    Combine all genre related tools and various cleaning methods to generate a 
    clean dataframe of genre with no overlappings between tag and merge_tags 
    column and within the merge_tags column.

- percentile
    Return a dataframe with subset of tags (descending order) of the input 
    dataframe which accounts for a certain percentage of the total counts of 
    all the tags.
    
- generate_vocal_txt
    Generate a txt file with a list of tags for each of the vocal tag filtered
    by percentile().

- generate_vocal_df
    Return a dataframe based on the manually-filtered txt files provided for
    each of the vocal tags.

- generate_final_csv
    Combine all the tools and generate the final dataframe consisting of 
    merging tags for each genre tag and vocal tag respectively.


'''
import pandas as pd
import numpy as np
import os
import sqlite3
import re

path = '/srv/data/msd/lastfm/SQLITE/lastfm_tags.db' # default path
#path = 'C://Users/hcw10/UROP2019/lastfm_tags.db'
filename = os.path.basename(path)[:-3]

output_path = '/srv/data/urop'
#output_path = 'C://Users/hcw10/UROP2019'


def set_path(new_path):
    ''' Set new_path as default path for the lastfm_tags.db database. '''
    global path
    global filename
    path = new_path
    filename = os.path.basename(path)[:-3]
  
    
    
def set_output_path(new_path):
    global output_path
    output_path = os.path.normpath(new_path)



def db_to_csv():
    
    #get all table names
    cnx = sqlite3.connect(path)
    cursor = cnx.cursor().execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    tables = [_[0] for _ in tables]
        
    #save everything for each table as individual csv
    for table in tables:
        
        csv_dir = os.path.join(output_path, filename +'_'+table+'.csv')
        print('saving '+ filename +'_'+table+'.csv')
        df = pd.read_sql_query("SELECT * FROM "+table, cnx)
        
        df.to_csv(csv_dir, index_label=False)
    
    
        
def insert_index(df):
    
    df['lastfm_ID']=np.arange(1, len(df)+1)
    df.set_index('lastfm_ID')
    return df



def popularity(csv_from_db=False):
    
    if csv_from_db:
        db_to_csv()
    
    #table names
    tables = ['tags', 'tid_tag']
    
    #load two of the csv saved
    filenames = [filename + '_' +table + '.csv' for table in tables]
    file_paths = [os.path.join(output_path, filename) for filename in filenames]
    tag = pd.read_csv(file_paths[0])
    tt = pd.read_csv(file_paths[1])
    
    #insert row_num from database
    tag = insert_index(tag)
    tag.set_index('lastfm_ID', inplace=True)
    
    #count number of occurence of each tag
    right = tt.tag.value_counts().to_frame()
    left = tag.tag.to_frame()
    
    popularity = left.merge(right, left_index=True, right_index=True)
    colnames = popularity.columns
    popularity.rename(columns={colnames[0]:'tags', colnames[1]:'counts'}, inplace=True)
    popularity = popularity.sort_values('counts', ascending=False)
    popularity['ranking'] = np.arange(1, len(popularity)+1)
    popularity.reset_index(inplace=True)
    popularity.set_index('ranking', inplace=True)
    popularity.rename(columns={'index':'lastfm_ID'}, inplace=True)
    
    return popularity

def clean_1(tag):
    tag_sub = re.sub(r'[^A-Za-z0-9]', '', tag)
    return tag_sub



def clean_2(tag):
    tag_sub = re.sub(r'[^A-Za-z0-9&]', '', tag)
    tag_sub = re.sub('&', 'n', tag_sub)
    return tag_sub


def clean_3(tag):
    tag_sub = re.sub(r'[^A-Za-z0-9&]', '', tag)
    tag_sub = re.sub('&', 'and', tag_sub)
    return tag_sub

def clean_4(tag):
    tag_sub = re.sub(' and ', 'n', tag)
    tag_sub = re.sub(r'[^A-Za-z0-9]', '', tag_sub)
    return tag_sub


def clean_5(tag):
    tag_sub = re.sub(r'[^A-Za-z0-9]', '', tag)
    tag_sub = re.sub(r'\b\d0s', lambda x:'19'+x.group(), tag_sub)
    return tag_sub


def clean_6(tag):
    tag_sub = re.sub(r'[^A-Za-z0-9]', '', tag)
    tag_sub = re.sub(r'\b\d0s', lambda x:'20'+x.group(), tag_sub)
    return tag_sub
    


def check_plural(x):
    if len(x)<2:
        return x
    
    elif (x[-1]=='s') & (x[-2]!='s'):
        return x[:-1]
    
    else:
        return x
    
    
    
def check_overlap(df_input):
    df = df_input.copy()
    
    ind = 1
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
                    df.iloc[idx].name
                    overlapped_idx_list.append(df.iloc[idx].name)
                    ind = 1
                    print(ind)
                    break
    
        #drop rows
        df = df.drop(overlapped_idx_list)
        
    return df
                

def combine_tags(df_input, list_of_tags, merge_idx=None):
    df = df_input.copy()
    
    list_of_tags = [item  if type(item)==str else str(item) for item in list_of_tags]
    
    #find the index of tag that all other tags will be merged to
    rows = df[df.tag.isin(list_of_tags)]
    
    if merge_idx:
        merge_index = merge_idx
    else:
        merge_index = np.min(rows.index.tolist())

    
    for idx in range(rows.shape[0]):
        #merge all rows whose row number is not the merge_index
        if rows.iloc[idx,:].name != merge_index:
            row = rows.iloc[idx,:]

            df.loc[merge_index, 'merge_tags'] = list(set(row.merge_tags+[row.tag]+df.loc[merge_index, 'merge_tags']))
            df = df.drop(row.name)
        
    return df



def add_tags(df_input, list_of_tags, target_tag, target_merge_index=True):
    df = df_input.copy()
    
    list_of_tags = [item  if type(item)==str else str(item) for item in list_of_tags]
    
    if type(target_tag) != str:
        target_tag = str(target_tag)
    
    if target_tag not in df.tag.tolist():
        #find index of the row the tag belongs to
        _ = [idx  for idx, row in enumerate(df.merge_tags) if target_tag in row][0]
        target_tag = df_input.iloc[_].tag
        
    
    if target_tag in list_of_tags:
        list_of_tags.remove(target_tag)
    
    common_tags = list(set(df.tag.tolist()).intersection(set(list_of_tags)))
    common_tags.append(target_tag)

    remain_tags = [tag for tag in list_of_tags if tag not in common_tags]

    
    target_list = df[df.tag==target_tag].iloc[0].merge_tags
    target_row = df[df.tag==target_tag]
    
    df.loc[df.tag==target_tag, 'merge_tags'] = [list(set(target_list+remain_tags))]
    
    if target_merge_index:
        merge_idx = target_row.index[0]
        df = combine_tags(df, common_tags, merge_idx)
    else:
        df = combine_tags(df, common_tags)
    
    df = check_overlap(df)
    
    return df

    
    
    
def remove_tag(df_input, tag):
    
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


def search_genre_exact(df_input, df_output, search_method=clean_1, threshold=2000, 
                       min_count=10, df_thr=None, verbose=True, remove_plural=True):
    
    df = df_input.copy()
    df['tags'] = df['tags'].astype('str')
    df['tags_sub'] = df.tags.apply(search_method)
    if remove_plural:
        df['tags_sub'] = df.tags_sub.apply(check_plural)
    
    if df_thr is None:
        df_thr = df[df.counts>=threshold]

    bool1 = df['counts']>=10
    
    for num, tag in enumerate(df_thr.tags):
        
        #remove non-word characters
        tag_sub = search_method(tag)
        if remove_plural:
            tag_sub = check_plural(tag_sub)
            
        #query
        search = r'^'+tag_sub+r'$'
            
        #get list of tags that will be merged (tag such that its 'pure' form 
        # matches with tag_sub)
        bool2 = df.tags_sub.str.findall(search, re.IGNORECASE).str.len()>0
            
        merge_tags = df[(bool1 & bool2)].tags.tolist()
        merge_tags = [item for item in merge_tags if item != tag] 
            
        df_output = add_tags(df_output, merge_tags, tag, False)
            
        if verbose:
            if (num+1)%10==0:
                print('processed {} tags'.format(num+1))
                
    return df_output
    


def merge_df(list_of_df):
    
    #function for combine elements in two columns for later use
    def row_op(row):
        col1, col2 = row.index[-2], row.index[-1]
        list1, list2 = row[col1], row[col2]
        if type(list1) != list:
            list1 = []

        if type(list2) != list:
            list2 = []

        return list(set(list1+list2))
    
    
    df = list_of_df[0].copy()
    
    for df2 in list_of_df[1:]:
        #outer merge and use row_op to combine the two list of merge_tags
        df_merge = df.merge(df2, how='outer', on='tag')
        df_merge['merge_tags'] = df_merge.apply(row_op, axis=1)
        df_merge = df_merge[['tag','merge_tags']]
        
        #search for tags that only exist in one of df and df2
        new_tags = []
        new_tags.append(list(set(df_merge.tag).difference(set(df.tag))))
        new_tags.append(list(set(df_merge.tag).difference(set(df2.tag))))
        new_tags = [i for sublist in new_tags for i in sublist]
        
        #make sure new standalone tags do not exist in any of themerge_tags lists
        #(there is no need to check for old tags because we have already ensured
        # that old common tags and all tags contained in merge_tags column is 
        # mutually exclusive)
        for tag in new_tags:
            for idx, l in enumerate(df_merge.merge_tags):
                if tag in l:
                    tag2 = df_merge.iloc[idx, :].tag
                    df_merge = combine_tags(df_merge, [tag, tag2], merge_idx=idx)
                    
                    break

    df_merge = check_overlap(df_merge)
    
    return df_merge


    
def generate_non_genre_droplist_txt(threshold=2000, csv_from_db=False):
    df = popularity(csv_from_db=csv_from_db)
    
    tag_list = df[df.counts>=threshold].tags.tolist()
    with open(os.path.join(output_path, 'non_genre_list.txt'), 'w', encoding='utf8') as f:
        for tag in tag_list:
            f.write("%s\n" % tag)


def generate_genre_df(csv_from_db=True, threshold=2000, min_count=10, verbose=True, 
                      drop_list_filename='non_genre_list_filtered.txt', indicator='-'):
    
    
    txt_path = os.path.join(output_path, drop_list_filename)
    drop_list = []
    with open(txt_path, 'r', encoding='utf8') as f:
        for item in f:
            if ((item[0]==indicator) and (not item.isspace()) and (item!='')):
                drop_list.append(item.rstrip('\n').lstrip(indicator))
                
    
    
    
    print('Genre-Step 1/8')
    df = popularity(csv_from_db=csv_from_db)
    #drop tags
    df.tags = df.tags.astype(str)
    df = df[-df.tags.isin(drop_list)]
    
    df_thr = df[df.counts>=threshold]
    df_output = pd.DataFrame({'tag':df_thr.tags.tolist(), 'merge_tags':[[]]*len(df_thr)})
    
    df_filter = search_genre_exact(df, df_output)
    
    print('Genre-Step 2/8')
    df_thr = df_thr[df_thr.tags.str.contains('&')]

    df_filter = search_genre_exact(df, df_filter, search_method=clean_2, threshold=threshold, 
                                   min_count=10, df_thr=df_thr, verbose=verbose)
    print('Genre-Step 3/8')
    df_filter = search_genre_exact(df, df_filter, search_method=clean_3, threshold=threshold,
                                   min_count=10, df_thr=df_thr, verbose=verbose)
    print('Genre-Step 4/8')
    df_thr = df[df.counts>=threshold]
    df_thr = df_thr[df_thr.tags.str.contains(' and ')]
    df_filter = search_genre_exact(df, df_filter, search_method=clean_4, threshold=threshold,
                                   min_count=10, df_thr=df_thr, verbose=verbose)
    print('Genre-Step 5/8')
    df_thr = df[df.counts>=threshold]
    df_thr = df_thr[df_thr.tags.str.contains(r'\b\d\ds')]
    df_filter =  search_genre_exact(df, df_filter, search_method=clean_5, threshold=threshold,
                                    min_count=10, df_thr=df_thr, remove_plural=False, verbose=verbose)
    
    print('Genre-Step 6/8')
    df_filter =  search_genre_exact(df, df_filter, search_method=clean_6, threshold=threshold,
                                    min_count=10, df_thr=df_thr, remove_plural=False, verbose=verbose)
    
    print('Genre-Step 7/8')
    
    return df_filter
    


def percentile(df, perc=0.9):
    tot_counts = df.counts.sum()
    threshold = perc * tot_counts
    counter = 0
    
    for i, row in enumerate(df.counts):
        if counter < threshold:
            counter += row
        else:
            return df.iloc[:i,]
    
    

def generate_vocal_txt(csv_from_db=False,
                       tag_list = ['female', 'instrumental', 'male', 'rap'],
                       perc_list=[0.9, 0.9, 0.9, 0.8]):
    
    df = popularity(csv_from_db=csv_from_db)
    
    def generate_txt(df, tag, perc):
        df_thr = df[df.tags.str.findall(r'\b'+tag, re.IGNORECASE).str.len()>0]
        df_thr = percentile(df_thr, perc=perc).tags.tolist()
        with open(os.path.join(output_path, tag+'_list.txt', encoding='utf8'), 'w') as f:
            for item in df_thr:
                f.write("%s\n" % item)
    
    if len(tag_list) != len(perc_list):
        print('length of tag_list is unequal to length of perc_list')
    
    for idx in range(len(tag_list)):
        generate_txt(df, tag_list[idx], perc_list[idx])

    print('Please deselect tags from generated txts by putting a "-" sign \
          the front of the term, or other symbol by adjusting the indicator \
          input variable in the generate_vocal_df() function.\n \
           E.g. If you want to deselect "female", put \
          "-female". Finally, please rename the files under the same \
          directory as the output files by adding a suffix _filtered. E.g.\
          save the filtered "female_list.txt" as "female_list_filtered.txt"')
    

def generate_vocal_df(indicator='-', 
                      tag_list = ['female', 'instrumental', 'male', 'rap']):
    
    def load_txt(filename):
        txt_path = os.path.join(output_path, filename)
    
        tag_name = re.findall(r'([A-Za-z0-9]+)_list', txt_path)[0]
        tags = []
        with open(txt_path, 'r', encoding='utf8') as f:
            for item in f:
                if ((item[0]!=indicator) and (not item.isspace()) and (item!='')):
                    tags.append(item.rstrip('\n'))
        return tags, tag_name 
    
    tag_list = []
    merge_tags_list=[]
    
    for filename in tag_list:
        tags, tag_name = load_txt(filename+'_list_filtered.txt')
        tag_list.append(tag_name)
        if filename in tags:
            tags.remove(filename)
        merge_tags_list.append(tags)
        
    
        
    df_filter =  pd.DataFrame({'tag':tag_list, 'merge_tags':merge_tags_list})
    return df_filter


def generate_final_csv(csv_from_db=True, threshold=2000, min_count=10, verbose=True,
                       pre_drop_list_filename='non_genre_list_filtered.txt',
                       combine_list=[['rhythm and blues', 'rnb']], 
                       drop_list=['2000', '00', '90', '80', '70', '60'],
                       add_list=None, add_target=None, add_target_merge_index=True):
    
    vocal = generate_vocal_df()
    genre = generate_genre_df(csv_from_db=csv_from_db, threshold=threshold,
                              min_count=min_count, verbose=verbose,
                              drop_list_filename=pre_drop_list_filename)
    
    df_final = pd.concat([genre, vocal])
    
    for item in combine_list:
        df_final = combine_tags(df_final, item)
        
    for item in drop_list:
        df_final = remove_tag(df_final, item)
    
    if add_list is not None:
        if len(add_list)!=len(add_target):
            print('length of add_list is unequal to length of add_target.')
        
        for idx in range(len(add_list)):
            if len(add_target_merge_index)==1:
                df_final = add_tags(df_final, add_list[idx], add_target[idx],
                                    target_merge_index=add_target_merge_index)
            
            if len(add_list)>1 and len(add_target_merge_index) == len(add_list):
                df_final = add_tags(df_final, add_list[idx], add_target[idx],
                                    target_merge_index=add_target_merge_index[idx])
                
            if len(add_list)!=len(add_target_merge_index):
                print('lenght of add_list is unequal to length of \
                      add_target_merge_index')
    
    df_final = df_final.reset_index(drop=True)
    
    return df_final




    




        
    
    
    
    
    
    
    
    
    
    
    
    
        




    
    
    
