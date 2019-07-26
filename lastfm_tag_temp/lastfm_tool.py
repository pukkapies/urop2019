import pandas as pd
import numpy as np
import os
import sqlite3
import re

path = '/srv/data/msd/lastfm/SQLITE/lastfm_tags.db' # default path
filename = os.path.basename(path)[:-3]
#path = 'C://Users/hcw10/UROP2019/lastfm_tags.db'
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


def percentile(df, perc=0.9):
    tot_counts = df.counts.sum()
    threshold = perc * tot_counts
    counter = 0
    
    for i, row in enumerate(df.counts):
        if counter < threshold:
            counter += row
        else:
            return df.iloc[:i,]





def check_plural(x):
    if len(x)<2:
        return x
    
    elif (x[-1]=='s') & (x[-2]!='s'):
        return x[:-1]
    
    else:
        return x


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
    

    

def search_genre_exact(df_input, search_method=clean_1, threshold=2000, min_count=10, verbose=True):
    
    df = df_input.copy()
    #append column with tags with non-word characters removed ('pure' form)
    df['tags'] = df['tags'].astype('str')
    df['tags_sub'] = df.tags.apply(search_method)
    df['tags_sub'] = df.tags_sub.apply(check_plural)
    
    df_thr = df[df.counts>=2000]
    
    #store tags that will be merged
    merge_tags_list = []
    
    #store tags that 
    useful_tags = []
    
    bool1 = df['counts']>=10
    
    for num, tag in enumerate(df_thr.tags):
        if not any(tag in sublist for sublist in merge_tags_list):
            
            #remove non-word characters
            tag_sub = search_method(tag)
            tag_sub = check_plural(tag_sub)
            
            #query
            search = r'^'+tag_sub+r'$'
            
            #get list of tags that will be merged (tag such that its 'pure' form 
            # matches with tag_sub)
            bool2 = df.tags_sub.str.findall(search, re.IGNORECASE).str.len()>0
            
            merge_tags = df[(bool1 & bool2)].tags.tolist()
            merge_tags = [item for item in merge_tags if item != tag]
            
            if len(set(merge_tags+useful_tags)) == (len(merge_tags) + len(useful_tags)):
                merge_tags_list.append(merge_tags)
                useful_tags.append(tag)
            else:
                print('overlapped')

                overlapped_tags = list(set(merge_tags).intersection(useful_tags))
                
                #tag name and position of the tag that will be used
                tag_use = overlapped_tags[0]
                pos_use = useful_tags.index(tag_use)
                for item in overlapped_tags[1:]:
                    
                    #position of item in useful_tags
                    pos = useful_tags.index(item)
                    
                    #the sub-list of merge_tags_list correspond to the item 
                    _ = merge_tags_list[pos]
                    
                    #remove the previous items from list
                    merge_tags_list.remove(_)
                    useful_tags.remove(item)
                    
                    #combine merge_tags
                    merge_tags = list(set(_ + merge_tags))
                    
                    
                merge_tags_list[pos_use] = list(set(merge_tags+merge_tags_list[pos_use]))
                
        
        if verbose:
            if (num+1)%10==0:
                print('processed {} tags'.format(num+1))
     
 
        
    
    df_genre = pd.DataFrame({'tag':useful_tags, 'merge_tags':merge_tags_list})
    
    #indicator    
    ind = 1
    #indicator=0 if no overlap between merge_tag sublists 
    while ind==1:
        df_genre, ind = check_overlap(df_genre)
        print(ind)
    
    return df_genre



def check_overlap(df_input):
    df = df_input.copy()
    overlapped_idx_list = []
    #indicator, if no overlap occurs, ind remains 0
    ind = 0
    
    for idx, l in enumerate(df['merge_tags']):
        for idx2, l2 in enumerate(df['merge_tags'].iloc[:idx]):
            #check overlap
            if  set(l2).intersection(l) != set():

                #merge tags to upper list
                df['merge_tags'].iloc[idx2] = list(set(l2+l))+[df['tag'].iloc[idx]]
                #clear tags in lower list so no overlap because of this list in the future
                df['merge_tags'].iloc[idx] = []
                overlapped_idx_list.append(idx)
                ind = 1
                break
    
    #drop rows
    df = df.drop(overlapped_idx_list)
    return df, ind
                
    
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
    
    
    df = list_of_df[0]
    
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
                    
                    
                    #r = df_merge[df_merge.tag==tag]
                    #_ = df.columns.tolist().index('merge_tags')
                    #df_merge.iloc[idx, _] = list(set(r.tag.tolist()+r.merge_tags.tolist()[0]+df_merge.iloc[idx, _]))
                    #df_merge = df_merge.drop(r.index[0])
                    break

        ind = 1
        while ind==1:
            df_merge, ind = check_overlap(df_merge)
            print(ind)
        df = df_merge.copy()
    
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
        if idx != merge_index:
            row = rows.iloc[idx,:]
            print(row)
            _ = df.columns.tolist().index('merge_tags')

            df.iloc[merge_index, _] = list(set(row.merge_tags+[row.tag]+df.iloc[merge_index, _]))
            df = df.drop(row.name)
        
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
        
                

        
    
    
    
    
    

def generate_csv():
    
    df = popularity()
    
    df1 = search_genre_exact(df, search_method=clean_1, 
                                   threshold=2000, min_count=10, verbose=True)
    
    df2 = search_genre_exact(df, search_method=clean_2, 
                                   threshold=2000, min_count=10, verbose=True)
    
    df3 = search_genre_exact(df, search_method=clean_1, 
                                   threshold=2000, min_count=10, verbose=True)
    
    df_output = merge_df([df1, df2, df3])
    
    df_output.to_csv(os.path.join(output_path, 'filtered_tag.csv'))
    
    

            
            
            
            
            
    













def load_txt(filename):
    txt_path = os.path.join(output_path, filename)
    
    tag_name = re.findall(r'([A-Za-z0-9]+)_list', txt_path)[0]
    tags = []
    with open(txt_path, 'r') as f:
        for item in f:
            if item[0] != '-':
                tags.append(item.rstrip('\n'))
    return tags, tag_name 
    


#def generate_tags_index(category, category_list):
#    '''give IDs to tags in each category'''
#    
#    ID_root = category_index[category]
#    category_ID = ID_root*100+np.arange(1, len(category_list)+1)
#    df = pd.Series(data = category_ID, index=category_list, name='ID')
#    return df


#txt_paths_list = []

#def generate_csv():
#    Ins_df = generate_tags_index('Instrument', Instrument_list)
#    Voc_df = generate_tags_index('Vocal', Instrument_list)
#    Gen_df = generate_tags_index('Genre', Genre_list)
#    Roc_df = generate_tags_index('Rock_sub', Rock_sub_list)
#    Hip_df = generate_tags_index('Hip_Hop_sub', Hip_Hop_sub_list)
#    Jaz_df = generate_tags_index('Jazz_sub', Jazz_sub_list)
#    clean_tags_df = pd.concat([Ins_df, Voc_df, Gen_df, Roc_df, Hip_df, Jaz_df]).to_frame()
#    clean_tags_df.rename(columns={clean_tags_df.columns[0]:'tag'}, inplace=True)
    
#    dirty_tags_list = []
#    dirty_df = pd.Series()
#    for txt_path in txt_paths_list:
#        dirty_tags, tag_name = load_txt(txt_path)
#        dirty_tags_list.append(dirty_tags)
        
        #convert list to df of dirty tag names and clean tag names
#        _ = pd.Series(data = [tag_name]*len(dirty_tags), index=dirty_tags)
#        dirty_df = pd.concat([dirty_df, _])
    
#    dirty_df = dirty_df.to_frame()
#    dirty_df.rename(columns={dirty_df.columns[0]:'tag'}, inplace=True)
    
    #dataframe with dirty tags used only
#    df = dirty_df.merge(clean_tags_df, on='tag')
    
    
    #get all tags
#    tag_path = os.path.join(output_path,  filename + '_' + 'tags' + '.csv')
#    tag_df = pd.read_csv(tag_path)
#    tag_df = insert_index(tag_df)
#    na_index = tag_df[tag_df.tag.isna()].index
#    tag_df = tag_df.dropna()
#    all_tags = tag_df.tag.to_list()
#    unused_tags = [t for t in all_tags if t not in dirty_tags]
#    
#    _ = len(unused_tags)
#    unused_df = pd.DataFrame({'tag':[None]*_, 'ID': np.zeros(_)})

#    df = pd.concat([df, unused_df])
    
    #merge with lastfm ID
#    df = df.merge(tag_df, on='tag')
    
#    for idx, na in enumerate(na_index):
#        na_row = {'tag':None, 'ID':0, 'lastfm_ID':na}
#        na_df = pd.DataFrame(data=na_row, index='Na'+str(idx))
#        ###
#        df = pd.concat([df, na_df])
    
#    return df

        
    
    
    
    
    
    
    
    
    
    
    
    
        




    
    
    
