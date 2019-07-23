import pandas as pd
import numpy as np
import os
import sqlite3
import re

path = '/srv/data/msd/lastfm/SQLITE/lastfm_tags.db' # default path
filename = os.path.basename(path)[:-3]
#path = 'C://Users/MacBook Pro/UROP2019/lastfm_tags.db'


def set_path(new_path):
    ''' Sets new_path as default path for the lastfm_tags database. '''
    global path
    path = new_path
    filename = os.path.basename(path)[:-3]

def db_to_csv(output_path=None):
    
    cnx = sqlite3.connect(path)
    cursor = cnx.cursor().execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    tables = [_[0] for _ in tables]
        

    for table in tables:
        
        csv_dir = os.path.join(os.path.normpath(output_path), filename +'_'+table+'.csv')
        print('saving '+ filename +'_'+table+'.csv')
        df = pd.read_sql_query("SELECT * FROM "+table, cnx)
        
        df.to_csv(csv_dir, index_label=False)
    
            

            
        
def insert_index(df):
    
    df['ID']=np.arange(1, len(df)+1)
    df.set_index('ID')
    return df


def popularity(csv_common_path):
    #table names
    tables = ['tags', 'tid_tag']
    
    #load two of the csv saved
    filenames = [filename + '_' +table + '.csv' for table in tables]
    file_paths = [os.path.join(os.path.normpath(csv_common_path), filename) for filename in filenames]
    tag = pd.read_csv(file_paths[0])
    tt = pd.read_csv(file_paths[1])

    tag = insert_index(tag)
    tag.set_index('ID', inplace=True)
    
    right = tt.tag.value_counts().to_frame()
    left = tag.tag.to_frame()
    
    popularity = left.merge(right, left_index=True, right_index=True)
    colnames = popularity.columns
    popularity.rename(columns={colnames[0]:'tags', colnames[1]:'counts'}, inplace=True)
    popularity = popularity.sort_values('counts', ascending=False)
    popularity['ranking'] = np.arange(1, len(popularity)+1)
    popularity.reset_index(inplace=True)
    popularity.set_index('ranking', inplace=True)
    popularity.rename(columns={'index':'ID'})
    
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



        
def txt_to_df(txt_path):
    tag_name = re.findall(r'([A-Za-z0-9]+)_list', txt_path)
    tags = []
    with open(txt_path, 'r') as f:
        for item in f:
            if item[0] != '-':
                tags.append(item.rstrip('\n'))
    return tags, tag_name 


    
    
    
