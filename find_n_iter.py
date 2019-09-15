import projectname_input

train_dataset = projectname_input.generate_datasets_from_dir('/srv/data/urop/tfrecords-log-mel-spectrogram', 'log-mel-spectrogram', sample_rate=16000, batch_size=128, split=(80, 10, 10))[0]

count = 0
for i in train_dataset:
    count += 1
    if count % 100 == 0:
        print(count)
print(count)
