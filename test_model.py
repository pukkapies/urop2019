import modules.query_lastfm as q_fm
import scripts.train as train

def test(model, tfrecord_dir, audio_format, train_val_test_split, batch_size, window_length, random, with_tags, with_tids):
    ''' Tests model '''

    test_datasets = \ 
    train.generate_datasets(tfrecord_dir=tfrecord_dir, 
                      train_val_test_split=train_val_test_split, 
                      which = [False, False, True],
                      batch_size=batch_size, shuffle=shuffle, 
                      buffer_size=buffer_size, window_length=window_length, 
                      random=random, with_tags=with_tags, with_tids=with_tids, 
                      num_epochs=1)

    AUC = tf.keras.metrics.AUC(name='AUC', dtype=tf.float32)

    for dataset in train_datasets:
        for entry in dataset:

            x_batch, y_batch = entry['audio'], entry['tags']

            logits = model(x_batch)

            AUC(y_batch, logits)

    print('Test set AUC: ', AUC.result())

def predict(model, audio, with_tags, db_path='/srv/data/urop/lastfm_clean.db')
    ''' Predicts tags given audio for one track '''

    logits = model(audio)
    fm = q_fm.LastFm(db_path)

    tags = []
    
    # if there is only one array
    if isinstance(audio[0], float):
        tags = []

        for idx, acc in enumerate(logits):
            if acc > 0.5:
                tags.append((fm.tag_num_to_tag(with_tags[idx-1]), acc))

        return tags
    else:
        track_tags = []

        for track_audio in audio:
            tags = []
            for idx, acc in enumerate(logits):
                if acc > 0.5:
                    tags.append((fm.tag_num_to_tag(with_tags[idx-1]), acc))
            
            track_tags.append(tags)

        return track_tags
