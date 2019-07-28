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
    ''' Sets new_path as default path for the lastfm_tags database. '''
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



def popularity():
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



def clean_2(tag, reverse=True):
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
    tag_sub = re.sub(r'[^A-Za-z0-9&]', '', tag)
    tag_sub = re.sub(r'\b\d\ds', lambda x:'19'+x.group(), tag_sub)
    return tag_sub


def clean_6(tag):
    tag_sub = re.sub(r'[^A-Za-z0-9&]', '', tag)
    tag_sub = re.sub(r'\b\d\ds', lambda x:'20'+x.group(), tag_sub)
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
        
        #make sure new tags do not exist in any merge_tags lists
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




def percentile(df, perc=0.9):
    tot_counts = df.counts.sum()
    threshold = perc * tot_counts
    counter = 0
    
    for i, row in enumerate(df.counts):
        if counter < threshold:
            counter += row
        else:
            return df.iloc[:i,]
    
     

def generate_genre_df(csv_from_db=True, threshold=2000, min_count=10, verbose=True):
    
    if csv_from_db:
        db_to_csv()
    
    print('Genre-Step 1/8')
    df= popularity()
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
    


def generate_vocal_txt(df, perc_list=[0.9, 0.9, 0.9, 0.8]):
    def generate_txt(df, tag, perc=0.9):
        df_thr = df[df.tags.str.findall(r'\b'+tag, re.IGNORECASE).str.len()>0]
        df_thr = percentile(df_thr, perc=perc).tags.tolist()
        with open(os.path.join(output_path, tag+'_list.txt'), 'w') as f:
            for item in df_thr:
                f.write("%s\n" % item)
    
    
    generate_txt(df, 'female', perc_list[0])
    generate_txt(df, 'instrumental', perc_list[1])
    generate_txt(df, 'male', perc_list[2])
    generate_txt(df, 'rap', perc_list[3])
    print('Please deselect tags from generated txts by putting a "-" sign at \
          the front of the term. E.g. If you want to deselect "female", put \
          "-female". Finally, please rename the files under the same \
          directory as the output files by adding a suffix _filtered. E.g.\
          save the filtered "female_list.txt" as "female_list_filtered.txt"')
    

def generate_vocal_df():
    def load_txt(filename):
        txt_path = os.path.join(output_path, filename)
    
        tag_name = re.findall(r'([A-Za-z0-9]+)_list', txt_path)[0]
        tags = []
        with open(txt_path, 'r') as f:
            for item in f:
                if ((item[0]!='-') and (not item.isspace()) and (item!='')):
                    tags.append(item.rstrip('\n'))
        return tags, tag_name 
    
    tag_list = []
    merge_tags_list=[]
    
    for filename in ['female', 'instrumental', 'male', 'rap']:
        tags, tag_name = load_txt(filename+'_list_filtered.txt')
        tag_list.append(tag_name)
        merge_tags_list.append(tags)
        
    df_filter =  pd.DataFrame({'tag':tag_list, 'merge_tags':merge_tags_list})
    return df_filter


def generate_final_csv(csv_from_db=True, threshold=2000, min_count=10, verbose=True):
    vocal = generate_vocal_df()
    genre = generate_genre_df(csv_from_db=csv_from_db, threshold=threshold,
                              min_count=min_count, verbose=verbose)
    
    df_final = pd.concat([genre, vocal])
    df_final = df_final.reset_index(drop=True)
    
    return df_final




    




        
    
    
    
    
    
    
    
    
    
    
    
    
        




    
    
    
