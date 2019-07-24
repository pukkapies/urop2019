import pandas as pd

# from aden import his module

def create_tid_tag_table(db: db.LastFm, df_tag_tag: pd.DataFrame):
    # fetch the tids from the original database, and map the tags to their correspondent 'clean' tags
    col_1 = db.tid_tag['tid']
    col_2 = db.tid_tag['tag'].apply(lambda tag: df_tag_tag['clean_tag'].loc[tag])

    # concatenate columns in a dataframe
    output = pd.concat([col_1, col_2], axis=1)

    # remove tags which fall below the popularity theshold
    output = output[output['tag'] != 0]
    return output

if __name__ = '__main__':
    
    # df = aden module .generate_df()
    df.reset_index(inplace=True)
    df.index = df.index + 1 # love pandas