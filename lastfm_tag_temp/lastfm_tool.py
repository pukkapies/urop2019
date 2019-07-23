import pandas as pd
import numpy as np
import os
import sqlite3
import re

path = '/srv/data/msd/lastfm/SQLITE/lastfm_tags.db' # default path
filename = os.path.basename(path)[:-3]
#path = 'C://Users/MacBook Pro/UROP2019/lastfm_tags.db'
output_path = '/srv/data/urop'


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
    
    cnx = sqlite3.connect(path)
    cursor = cnx.cursor().execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    tables = [_[0] for _ in tables]
        

    for table in tables:
        
        csv_dir = os.path.join(output_path, filename +'_'+table+'.csv')
        print('saving '+ filename +'_'+table+'.csv')
        df = pd.read_sql_query("SELECT * FROM "+table, cnx)
        
        df.to_csv(csv_dir, index_label=False)
    
            

            
        
def insert_index(df):
    
    df['ID']=np.arange(1, len(df)+1)
    df.set_index('ID')
    return df


def popularity():
    #table names
    tables = ['tags', 'tid_tag']
    
    #load two of the csv saved
    filenames = [filename + '_' +table + '.csv' for table in tables]
    file_paths = [os.path.join(output_path, filename) for filename in filenames]
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
    popularity.rename(columns={'index':'ID'}, inplace=True)
    
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






category_index = {'Instrument':1, 'Vocal':2, 'Genre':3, 
                  'Rock_sub':4, 'Hip_Hop_sub':5, 'Jazz_sub':6}

Instrument_list = []
Vocal_list = []
Genre_list = []
Rock_sub_list = []
Hip_Hop_sub_list = []
Jazz_sub_list = []


def load_txt(filename):
    txt_path = os.path.join(output_path, filename)
    
    tag_name = re.findall(r'([A-Za-z0-9]+)_list', txt_path)[0]
    tags = []
    with open(txt_path, 'r') as f:
        for item in f:
            if item[0] != '-':
                tags.append(item.rstrip('\n'))
    return tags, tag_name 
    


def generate_tags_index(category, category_list):
    '''give IDs to tags in each category'''
    
    ID_root = category_index[category]
    category_ID = ID_root*100+np.arange(1, len(category_list)+1)
    df = pd.Series(data = category_ID, index=category_list)
    return df


txt_paths_list = []

def generate_csv():
    Ins_df = generate_tags_index('Instrument', Instrument_list)
    Voc_df = generate_tags_index('Vocal', Instrument_list)
    Gen_df = generate_tags_index('Genre', Genre_list)
    Roc_df = generate_tags_index('Rock_sub', Rock_sub_list)
    Hip_df = generate_tags_index('Hip_Hop_sub', Hip_Hop_sub_list)
    Jaz_df = generate_tags_index('Jazz_sub', Jazz_sub_list)
    clean_tags_df = pd.concat([Ins_df, Voc_df, Gen_df, Roc_df, Hip_df, Jaz_df]).to_frame()
    clean_tags_df.rename(columns={clean_tags_df.columns[0]:'tag'}, inplace=True)
    
    dirty_df = pd.Series()
    for txt_path in txt_paths_list:
        dirty_tags, tag_name = load_txt(txt_path)
        
        #convert list to df of dirty tag names and clean tag names
        _ = pd.Series(data = [tag_name]*len(dirty_tags), index=dirty_tags)
        dirty_df = pd.concat([dirty_df, _])
    
    dirty_df = dirty_df.to_frame()
    dirty_df.rename(columns={dirty_df.columns[0]:'tag'}, inplace=True)
    
    df = dirty_df.merge(clean_tags_df, on='tag')
    
    return df


        
    
    
    
    
    
    
    
    
    
    
    
    
        




    
    
    
