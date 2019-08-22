import modules.query_lastfm as q_fm
import projectname_input

def test(model, tfrecord_dir, audio_format, config_dir, split, batch_size=64, window_length=15, random=False, with_tags=None, with_tids=None, merge_tags=None, num_tags=155):
    ''' Tests model '''

    if not os.path.isfile(config_dir):
        config_dir = os.path.join(os.path.normpath(config_dir), 'config.json')

    with open(config_dir) as f:
        file = json.load(f)

    num_tags = file['dataset_specs']['n_tags']

    test_datasets = \ 
    projectname_input.generate_datasets_from_dir(tfrecord_dir, audio_fromat, split=split, 
                      batch_size=batch_size, shuffle=False, window_size=window_size,
                      random=random, with_tags=with_tags, with_tids=with_tids,
                      merge_tags=merge_tags, num_tags=num_tags, num_epochs=1)[-1]

    ROC_AUC = tf.keras.metrics.AUC(curve='ROC', name='ROC_AUC',  dtype=tf.float32)
    PR_AUC = tf.keras.metrics.AUC(curve='PR', name='PR_AUC', dtype=tf.float32)

    for entry in dataset:

        audio_batch, label_batch = entry[0], entry[1]

        logits = model(audio_batch)

        ROC_AUC.update_state(label_batch, logits)
        PR_AUC.update_state(label_batch, logits)

    print('ROC_AUC: ', ROC_AUC.result(), '; PR_AUC: ', PR_AUC.result())

def predict(model, audio, with_tags, db_path='/srv/data/urop/lastfm_clean.db')
    ''' Predicts tags given audio for one track '''

    logits = model(audio)
    fm = q_fm.LastFm(db_path)

    tags = []
    
    # if there is only one array
    if isinstance(audio[0], float):
        tags = []

        for idx, probability in enumerate(logits):
            if probability > 0.5:
                tags.append((fm.tag_num_to_tag(with_tags[idx-1]), probability))

        return tags
    else:
        track_tags = []

        for track_audio in audio:
            tags = []
            for idx, probability in enumerate(logits):
                if probability > 0.5:
                    tags.append((fm.tag_num_to_tag(with_tags[idx-1]), probability))
            
            track_tags.append(tags)

        return track_tags

if __name__ == '__main__':
    CONFIG_FOLDER  = '/home/calle'
    main('', '',)
