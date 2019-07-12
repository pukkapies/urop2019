if __name__ == '__main__':
    # convert the (desired columns in the) HDF5 summary file as a dataframe
    df_summary = extract_ids_from_summary(PATH_TO_H5)
        
    # search for MP3 tracks through the MP3_ROOT_DIR folder
    df = find_tracks_with_7dids(MP3_ROOT_DIR)
    
    # create a new dataframe with the metadata for the tracks we actually have on the server
    df = df_merge(df_summary, df)
        
    # discard mismatches
    df = df_purge_mismatches(df, PATH_TO_MISMATCHES_TXT)
    
    # discard duplicates
    # df = df_purge_duplicates(df, PATH_TO_DUPLICATES_TXT)

    # save output
    output = 'ultimate_csv.csv'
    output_path = os.path.join(OUTPUT_DIR, output)

    our_df.to_csv(output_path, header=False, index=False)