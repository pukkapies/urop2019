import pandas as pd

# from aden import his module

def flatten(df: pd.DataFrame):
    tags =[]
    tags_nums = []

    for num, tag in enumerate(df['tags']):
        tags.append(tag)
        tags_nums.append(num)

        for tag in df['merged_tags'].iloc[num]:
            tags.append(tag)
            tags_num.append(num)

def create_tag_tag_table(db: db.LastFm, df: pd.DataFrame):
    flat = flatten_num(df)
    
    # fetch the tag num's from the original database
    old_tags = db.tag.index.to_series()

    # for each tag num, get the corresponding idx in the flattened dataframe (returns idx = 0 if tag falls below the pop threshold)
    new_tags = old_tags.apply(lambda t: flat['old_lastfm_tag'][flat['old_lastfm_tag'] == t].append(pd.Series([0])).index[0])
    
    # for each tag idx, get the corresponding 'clean' tag num from the flattened dataframe (returns tag num = 0 if tag falls below the pop threshold)
    new_tags = new_tags.apply(lambda i: flat['new_tag'].get(i))

    output = pd.DataFrame(data={'old_lastfm_tag': new_tags}, index=old_tags.rename('new_tag'))

def create_tid_tag_table(db: db.LastFm, df_tag_tag: pd.DataFrame):
    # fetch the tids from the original database, and map the tags to their correspondent 'clean' tags
    col_1 = db.tid_tag['tid']
    col_2 = db.tid_tag['tag'].apply(lambda t: df_tag_tag['clean_tag'].loc[t])

    # concatenate columns in a dataframe
    output = pd.concat([col_1, col_2], axis=1)

    # remove tags which fall below the popularity theshold
    output = output[output['tag'] != 0]
    return output

if __name__ = '__main__':
    
    # df = aden module .generate_df()
    df.reset_index(inplace=True)
    df.index = df.index + 1 # love pandas