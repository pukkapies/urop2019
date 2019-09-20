
# Deep Learning for Music Tagging (aka 'Orpheus')

This is the repository of an Imperial College UROP 2019 project in deep learning for music tagging. We aimed at developing an end-to-end music audio auto-tagger competitive with the state-of-the-art. We replicated the CNN architecture proposed by Pons et al. in [this]([https://arxiv.org/pdf/1711.02520.pdf](https://arxiv.org/pdf/1711.02520.pdf)) paper, and successfully reproduced the results they obtained with the Million Songs Dataset. Since our model learnt to predict some audio features quite accurately, we decided to call it 'Orpheus', like the legendary ancient Greek poet and musician.

## Table of Contents

* [Introduction](https://github.com/pukkapies/urop2019#introduction)
* [System Requirements](https://github.com/pukkapies/urop2019#system-requirements)
* [Data Cleaning](https://github.com/pukkapies/urop2019#data-cleaning)
    * [Errors in the .MP3 Audio Files](https://github.com/pukkapies/urop2019#audio)
    * [Errors in the Dataset](https://github.com/pukkapies/urop2019#database)
    * [Last.fm Tags](https://github.com/pukkapies/urop2019#tags)
* [Data Input Pipeline](https://github.com/pukkapies/urop2019#data-input-pipeline)
    * [TFRecords](https://github.com/pukkapies/urop2019#tfrecords)
    * [TFRecords into a tf.data.Dataset](https://github.com/pukkapies/urop2019#dataset-preparation)
* [Model and JSON Configuration](https://github.com/pukkapies/urop2019#dataset-preparation)
* [Training](https://github.com/pukkapies/urop2019#training)
* [Validating & Predicting](https://github.com/pukkapies/urop2019#evaluation-tools)
* [Results](https://github.com/pukkapies/urop2019#results)
* [References](https://github.com/pukkapies/urop2019#references)
* [Contacts / Getting Help](https://github.com/pukkapies/urop2019#contacts--getting-help)


## Introduction

This project makes use of the freely-available [Million Song Dataset]( http://millionsongdataset.com), and its integration with the [Last.fm](http://millionsongdataset.com/lastfm/) dataset. The former provides a link between all the useful information about the tracks (such as title, artist or year) and the audio track themselves, whereas the latter contains tags information on some of the tracks. A  preview of the audio tracks can be fetched from services such as 7Digital, but this is allegedly not an easy task. 

If you are only interested in our final result, click [here](https://github.com/pukkapies/urop2019#results).

If you are only interested in the 'lite' version of our prediction tool, click [here](TODO)

Otherwise, if you would to use some of our code, or try to re-train our model on your own, read on. We will assume you somehow have access to the actual audio files. Here is the outline of the approach we followed:

1. we extracted all the useful information from the Million Song Dataset and cleaned both the audio tracks and the Last.fm tags database to produce final 'clean' dataset;

2. we prepared the data input pipeline and transformed the data in a format which is easy to consume by the training algorithm;

3. we wrote a flexible training script which allowed for multiple custom experiments (such as slightly different architectures, slightly different versions of the tags database, or different training parameters);

4.  we trained our model and provided tools to make sensible tag prediction from a given input audio.

In the following sections, we will guide you through what we have done more in detail, and provide a brief tutorial of how you may use this repository to make genre predictions of your own, or carry out some further experiments.

## System Requirements

* [Python](https://www.python.org/)* 3.6 or above
* One or more CUDA-enabled GPUs
* Mac or Linux environment
* [TensorFlow](https://www.tensorflow.org/beta)* 2.0.0 RC0 or above (GPU version)
* [H5Py](https://www.h5py.org/) 2.3.1 -- to read the the HDF5 MSD summary 
* [LibROSA](https://github.com/librosa/librosa)* 0.7.0 + [FFmpeg](https://www.ffmpeg.org/)* -- to read, load and analyse audio files
* [mutagen](https://mutagen.readthedocs.io/en/latest/) 1.42.0 -- to read audio files
* [sounddevice](https://python-sounddevice.readthedocs.io/en/0.3.12/installation.html)* 0.3.12 -- to record audio from your microphone through terminal
* [sparse](https://sparse.pydata.org/en/latest/) 0.8.9 -- to perform advanced operations on the tags database (and process data using sparse matrices)
* [tqdm](https://github.com/tqdm/tqdm) 4.36.1 -- to display fancy progress bars
* Other common Python libraries such as [Pandas](https://pandas.pydata.org/) or [NumPy](https://numpy.org/)

If you are just running the lite version of our prediction tool, all you need are the packages marked with *.

## Data Cleaning

### Errors in the .MP3 Audio Files
Firstly, you can use `fetcher.py` to scan the directory which contains all the audio files, and store file path, file size, duration and number of channels in a Pandas dataframe. The audio files that cannot be opened correctly (that is, which have file size zero or duration zero) are removed.  

After that, you can use `mp3_to_numpy.py` to convert every working audio file into numpy arrays (one per audio channel). The numpy arrays of each track are then analysed using LibROSA in order to extract the location of any silent sections (yes, some tracks _can_ be opened, but they are completely silent). The information on silent sections, the audio arrays and the sample rate of each track can be optionally stored into .npz files. 

During the project, we created .npz files for the entire dataset as we did not have the whole audio cleaning path outlined yet, and we needed to have the data easily accessible for frequent experimentations. However, we would recommend against creating this approach, as creating .npz files often requires a huge amounts of storage space (MP3 is a compressed audio format for a reason...).

Finally, you can use `wrangler_silence.py` to process and interpret the information on audio silence, and append the results (e.g. effective_clip_length, or max_silence_length) to the original Pandas dataframe. The audio files which do not satisfy certain user-set criteria (such as a low percentage of silence) are removed.

*Example:*

This will create a .csv that contains the information of tracks mentioned above.

```
python fetch.py /srv/data/urop2019/fetch.csv --root-dir /srv/data/msd/7digital
```

This will generate npz files mentioned above.

```
python mp3_to_numpy.py /srv/data/urop2019/fetch.csv --root-dir-npz /srv/data/urop2019/npz --root-dir-mp3 /srv/data/msd/7digital
```

This will expand the fetch.csv generated above to include some interpretations 
of the silent information.

```
python wrangler_silence.py /srv/data/urop2019/fetch.csv /srv/data/urop2019/wrangle_silence.csv --root-dir-npz /srv/data/urop2019/npz --root-dir-mp3 /srv/data/msd/7digital --min-size 100000 --filter-tot-silence 15 --filter-max-silence 3
```

### Errors in the Dataset 
The raw HDF5 Million Song Dataset file, which contains three smaller datasets, 
are converted into multiple Pandas dataframes. The relevant information is then 
extracted and merged. According to the MSD website, there are mismatches between these datasets. For more details, see [here](http://millionsongdataset.com/blog/12-2-12-fixing-matching-errors/). To deal with this issue, `wrangler.py` takes a '.txt' file with a list of tids which could not be trusted, and remove the corresponding rows in the dataframe. Furthermore, MSD also provides a `.txt` file with a list of tracks that have duplicates. `wrangler.py` by default 
keeps one version of the duplicate tracks of each song and removes the rest.

The dataframe from the above paragraph is merged with the dataframe produced by the above audio section followed by removing unnecessary columns to produce the 'ultimate' dataframe. This dataframe acts as a clean dataset containing all the essential information about the tracks and will be used throughout the project.

For more information about how these functions are used, see [here](https://github.com/pukkapies/urop2019/blob/master/code/msd/README.md)

*Example:*

```
python wrangle.py /srv/data/urop/wrangle_silence.csv /srv/data/urop/ultimate.csv --path-h5 /srv/data/msd/entp/msd_summary_file.h5 --path-db /srv/data/msd/lastfm/lastfm_tags.db --path-txt-dupl /path/to/duplicates.txt --path-txt-mism /path/to/mismatches.txt
```

Alternatively, to save storage space and time, the following order of code 
execution was used instead:

```
python fetch.py /srv/data/urop2019/fetch.csv --root-dir /srv/data/msd/7digital
```

```
python wrangle.py /srv/data/urop2019/fetch.csv /srv/data/urop2019/fetch2.csv --path-h5 /srv/data/msd/entp/msd_summary_file.h5 --path-db /srv/data/msd/lastfm/SQLITE/lastfm_tags.db --path-txt-dupl /path/to/duplicates.txt --path-txt-mism /path/to/mismatches.txt --discard-dupl False
```

```
python mp3_to_numpy.py /srv/data/urop2019/fetch2.csv --root-dir-npz /srv/data/urop2019/npz --root-dir-mp3 /srv/data/msd/7digital
```

```
python wrangler_silence.py /srv/data/urop2019/fetch2.csv /srv/data/urop2019/wrangle_silence.csv --root-dir-npz /srv/data/urop2019/npz --root-dir-mp3 /srv/data/msd/7digital --min-size ? --filter-tot-silence 15 --filter-max-silence ?
```

```
python wrangle.py /srv/data/urop2019/wrangle_silence.csv /srv/data/urop2019/ultimate.csv --path-h5 /srv/data/msd/entp/msd_summary_file.h5 --path-db /srv/data/msd/lastfm/SQLITE/lastfm_tags.db --path-txt-dupl ???? --path-txt-mism /srv/data/msd/sid_mismatches.txt
```

With this order of execution, `wrangle.py` will remove tracks which 
have no tags. This reduces the number of tracks from 1,000,000 to 500,000+.

For more information on how you can customise the procedures, 
see the documentation in the corresponding scripts.

### Last.fm Tags
#### Performing Queries
The `lastfm.py` module contains two classes, `LastFm` and `LastFm2Pandas`, and each of them contains all the basic tools for querying the Lastfm database. The former directly queries the database by SQL, whereas the latter converts the database into .csv files and queries the data using Pandas. In some of the functions in latter sections, it may have a `lastfm` input parameter and require to be set as an instance of one of the classes. 

*Example:*

To use `LastFm`,

```python
fm = lastfm.LastFm('/srv/data/msd/lastfm/SQLITE/lastfm_tags.db')
```
To use `LastFm2Pandas`,

```python
fm = lastfm.LastFm2Pandas('/srv/data/msd/lastfm/SQLITE/lastfm_tags.db')
```
To use `LastFm2Pandas` from converted .csv's (instead of the original .db file),

```python
# generate .csv's
lastfm.LastFm('/srv/data/msd/lastfm/lastfm_tags.db').db_to_csv(output_dir='/srv/data/urop')
# create tags, tids and tid_tag dataframes
tags = pd.read_csv('/srv/data/urop/lastfm_tags.csv')
tids = pd.read_csv('/srv/data/urop/lastfm_tids.csv')
tid_tag = pd.read_csv('/srv/data/urop/lastfm_tid_tag.csv')
# create class instance
fm = lastfm.LastFm2Pandas.load_from(tags=tags, tids=tids, tid_tag=tid_tag)
```

Note that the major difference between the two classes is that 
`LastFm` is quicker to initiate, but some queries might take some time to 
perform, whereas `LastFm2Pandas` may take longer to initiate due to the whole 
dataset being loaded to the memory. However, it contains some more advanced methods, and it is quick to initialise if database is converted into .csv files in advance.


Finally, in order to explore the Million Song Dataset summary file, `metadata.py` contains some basic tools to query the `msd_summary_file.h5` file. 

#### Filtering

In the Last.fm database there are more than 500,000 different tags. Such a high number of tags is clearly useless, and the tags need to be cleaned in order for the training algorithm to learn to make some sensible predictions. The tags are cleaned using `lastfm_cleaning_utils.py` and `lastfm_cleaning.py`, and the exact mechanisms of how they work can be found in the documentation of the scripts.

In brief, the tags are divided into two categories: genre tags, and vocal tags (which in our case are 'male vocalist', 'female vocalist', 'rap' and 'instrumental'). For genre tags, we first obtained a list of tags from the Last.fm database which have appeared for more than 2000 times, then we manually filtered out the tags that we considered rubbish (such as 'cool' or 'favourite song'). We fed the remaining 'good' tags into `generate_genre_df()`, which searched for similar spelling of the same tag within a 500,000+ tags pool (tags with occurrence ≥ 10). We generated a handy dataframe with manually chosen tags in one column, and similar matching tags from the pool in the other.  For vocal tags, we obtained a long list of potentially matching tags for each of the four vocal tags, then we manually seperated the 'real' matching tags from the rest, for each of the tag lists. We fed the tag lists into `generate_vocal_df()`, and we produced a dataframe with structure previously outlined. Finally, the function `generate_final_df()` merged the two dataframes into one, and passed it to the next script.

To search for similar tags, we did the following:
1. Remove all the non-alphabet and non-number characters and any single trailing 's' 
from the raw tags with occurance ≥ 10 and the target tags (the classified genre tags 
and vocal tags). If any transformed raw tag is identical to any target tag, 
the raw tag is merged into target tag.

2. Repeat the same merging mechanism as 1, but replace '&' with 'n', '&' with 'and', ' n '
with 'and' instead respectively.

3. Repeat the same merging mechanism as 1, but replace any 'x0s' 
string with '19x0s', 'x0s' with '20x0' (x denodes a number character) without 
removing the trailing 's' respectively.

See [here](https://github.com/pukkapies/urop2019/tree/master/code/msd#tags-cleaning) for 
how you may tailor the merging mechanism by defining a new fitlering fucntion.

The `.txt` files containing the lists of tags we used in our experiment can be found in 
the folder `~/msd/config`. Hence, if you prefer to use our dataset, you may simply 
generate this by:

```python
generate_final_df(from_csv_path='/srv/data/urop', threshold=2000, sub_threshold=10, combine_list=[['rhythm and blues', 'rnb'], ['funky', 'funk']], drop_list=['2000', '00', '90', '80', '70', '60'])
```

if you are interested to view the dataset. Otherwise, `lastfm_clean.py` will automatically 
generate this dataset and transform it into a clean Lastfm database. 

Note that `lastfm_cleaning_utils` allows a great deal of customisation. 
Please see [here](https://github.com/pukkapies/urop2019/tree/master/code/msd#tags-cleaning)
 for more details.

`lastfm_cleaning.py` creates a new database file using the cleaned tags 
from lastfm_cleaning_utils.py. The database has the same structure as the 
`lastfm_tags.db` database, and can be queried by `lastfm.py`.

*Example:*
```
python lastfm_cleaning.py /srv/data/msd/lastfm/SQLITE/lastfm_tags.db /srv/data/urop/clean_lastfm.db --val ?? 
```

## Data Input Pipeline
### TFRecords
To store the necessary information we need in training, we used the `.tfrecord` 
file format. The `preprocessing.py` script does exactly this. In each entry of 
the `.tfrecord` file, it stores the audio as an array in either 
waveform or log mel-spectrogram format. It will also store the TID to identify each 
track as well as the tags from the `clean_lastfm.db` database in a one-hot vector format.

`preprocessing.py` leaves quite a lot of room for user customisation. It will 
accept audio as `.mp3` files, or as `.npz` files where each entry contains the 
audio as an array and the sample rate. The user can choose the sample rate to 
store the data in as well as the number of mel bins when storing the audios in 
the log mel-spectrogram form. It is also possible to specify the number of  `.tfrecord` files to split the data between.

In our case, we used 96 mel bins, a sample rate of 16kHz and split the data 
into 100 .`tfrecord` files. We also had the data stored as `.npz` files, since 
we have loaded the `.mp3` files as numpy for silence analysis and stored them 
in a previous section. However, we would recommend users to convert directly 
from `.mp3` files as the `.npz` files need a lot of storage. 

Example:

```
python preprocessing.py waveform /srv/data/urop/tfrecords-waveform --root-dir /srv/data/urop2019/npz --tag-path /srv/data/urop/clean_lastfm.db --csv-path /srv/data/urop/ultimate.csv --sr 16000 --num-files 100 --start-stop 1 1
```

Note that it is recommended to use tmux split screens to speed up the process.

### TFRecords into a tf.data.Dataset
`projectname_input.py` was used to create ready-to-use TensorFlow datasets 
from the `.tfrecord` files. Its main feature is to create 3 datasets for 
train/val/test by parsing the `.tfrecord` files and extracting a 15s window 
of the audio and then normalizing the data. If waveform is used, the normalization 
is simple batch normalization, but if log mel-spectrogram  is used, we normalized 
with respect to the spectrograms themselves (Pons, et al., 2018). The file will also create 
mini-batches of a chosen size.

Again we have left a lot of room for customization. There are functions to exclude 
certain TIDs from the dataset, to merge certain tags, e.g. rap and hip hop, and a 
function to only include some tags. The user can also choose the size of the windows 
mentioned above and whether they are to be extracted from a random position or centred 
on the audio array. 

Note that the data input pipeline is optimised following 
the [official guideline](https://www.tensorflow.org/beta/guide/data_performance) 
from TensorFlow 2.0.

Datasets will automatically be input to the training algorithm. To manually generate 
a dataset from one or more tfrecord files, you may use the generate_datasets() function 
in projectname_input.py

*Example:*

If you want to create a train, a validation dataset from one tfrecord file 
respectively, with top 10 tags from the popularity dataset based on the new 
`clean_lastfm.db` (ranking before tags merge), with 'pop', and 'alternative' 
merged, this is how you may do it,

```python
import projectname_input
import lastfm

#get top ten tags
lf = lastfm.LastFm('/srv/data/urop/clean_lastfm.db')
top_tags = lf.popularity()['tags'][:10].tolist()

#create datasets
tfrecords = ['/srv/data/urop/tfrecords-waveform/waveform_1.tfrecord', '/srv/data/urop/tfrecords-waveform/waveform_2.tfrecord']
train_dataset, valid_dataset = projectname_input.generate_datasets(tfrecords, audio_format='waveform', split=[1,1,0], which=[True, True, False], with_tags=top_tags, merge_tags=['pop', 'alternative'])
```

## Model and Configuration


The model we used was designed by (Pons, et al., 2018). See 
[here](https://github.com/jordipons/music-audio-tagging-at-scale-models) for more details. 
In our experiment, as mentioned above, we have followed (Pons, et al., 2018) and convert the audio files 
into **waveform** and **log mel-spectrogram** respectively for training. Since the model pipeline 
written by (Pons, et al., 2018) is only compatible with TensorFlow 1.x, we have rewritten the model with TensorFlow 2.0 
syntax in `projectname.py`.

In brief, `projectname.py` contains a frontend for waveform and log mel-spectrogram respectively and a backend model. 
The `build_model()` function combines a frontend and the backend to produce a complete neural network that will 
be used in the training algorithm.

In the training phase, parameters are supplied by a json file, instead of being specified as input parameters of 
the training loop functions. You may generate the json file with the default parameters (
from our experiment with training parameters suggested by (Pons, et al., 2018)) using the `create_config_json` function.

*Example:*

```python
projectname.create_config_json('/srv/data/urop/config.json')
```

A list of available parameters and their properties can be found in the documentation within 'projectname.py'. 
In short, it contains five categories of parameters:

1. **model**: any network-related parameters.

2. **optimizer**: name and learning rate of the optimiser.

3. **tags**: customised tags for dataset input pipeline.

4. **tfrecords**: parameters used when generating the tfrecords.

5. **config**: any parameters related to the training algorithm and the dataset input pipeline.

If you wish to change any parameters, e.g. change learning rate to 0.005 and batch 
size to 32, you may simply do:

```python
projectname.create_config_json('/srv/data/urop/config2.json', 'learning_rate'=0.005, 'batch_size'=32)
```

## Training

We have written two separate scripts for the training algorithm, `training.py` 
and `'training_gradtape.py`. The major difference between the two is that 
the former uses `model.fit` in Keras, whereas the latter contains a custom 
training loop which performs optimisation by `tf.GradientTape()`. Therefore, 
if you want to implement some special features from Keras such as callbacks, 
you may easily amend the codes of `training.py` to achieve that. Otherwise, 
since the training loop in `training_gradtape.py` is produced from scratch 
(it only relies on `tf.GradientTape()`, you may edit the codes of the function 
`train()`  to implement more advanced features. 

If you simply want to train the model with default settings, both of the scripts 
would work.

Note that both scripts use MirroredStrategy and assume you have one or more GPUs 
available. On the other hand, both scripts use the TensorBoard and checkpoints, and 
early stopping can be optionally enabled. The training algorithm in 'training.py' 
also contains the Keras callback ReduceLROnPlateau as an input option.

*Example:*

To perform a simple training with default settings in waveform for ten epochs on GPU 0,1, 

```
python training.py waveform, --root-dir /srv/data/urop/tfrecords-waveform --config-path /srvdata/urop/config.json --lastfm-path /srv/data/urop/clean_lastfm.db --epochs 10 --cuda 0 1
```
You may control all the parameters within the config.json file.

If you prefer to use `training_gradtape.py`, it works the same way as above to 
start the training.

Furthermore, it is possible to stop the scripts in the middle of training by keyboard interrupt
and recover from the last epoch (works for both scripts). Please refer to the documentation 
of the corresponding script for more details on how to do this with 
the `--resume-time` parameter. 

If you want to perform the model training with more flexibility in choosing 
the training dataset and validation dataset, you may follow the instruction on 
data input pipeline to generate the datasets and do the following:

```python
import os
import tensorflow as tf
import training
# initiate strategy
strategy = tf.distribute.MirroredStrategy()

# wrap the datasets
train_dataset = strategy.experimental_distribute_dataset(train_dataset)
valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)

#set GPUs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

#parse config
config, config_optim = training.parse_config('/srv/data/urop/config.json', '/srv/data/urop/clean_lastfm.db')

# train
training.train(train_dataset, valid_dataset, frontend='waveform', strategy=strategy, config=config, config_optim=config_optim, epochs=10)
```
If you prefer to use `training_gradtape.py`, do exactly the same procedure as above
except replacing `training` with `training_gradtape`.


## Validating and Predicting

`test_model.py` is the script containing the evaluation tools. There is a `test_model()` function which 
simply tests the model's performance on the test dataset from a certain checkpoint. The `predict()` function 
takes an audio array, in the waveform or the log mel-spectrogram format, and uses the model on consecutive 15s 
windows (with the last window 15s from the end of the track) of the input audio to return the 
average prediction as tags in string form.

*Example:*

To test the model on the last 10% of the tfrecord files on log mel-spectrogrm:

```
python log-mel-spectrogram test /srv/data/urop/config.json --checkpoint /srv/data/urop/model/log-mel-spectrogram_190826-103644/epoch-18 --lastfm-path /srv/data/urop/clean_lastfm.db --tfrecords-dir /srv/data/urop/tfrecords-log-mel-spectrogram
```

To make prediction to an audio file and display tags with minimum score 0.1:

```
python log-mel-spectrogram predict /srv/data/urop/config.json --checkpoint /srv/data/urop/model/log-mel-spectrogram_190826-103644/epoch-18 --lastfm-path /srv/data/urop/clean_lastfm.db --mp3-path /srv/data/urop/song.mp3 --cutoff 0.2
```

If you have a directory which contains only audio files (one or more), you may set `--mp3-path` as the directory path.

To make prediction by recording a 30s audio with your microphone in terminal:

```
python test_model.pylog-mel-spectrogram predict /srv/data/urop/config.json --checkpoint /srv/data/urop/model/log-mel-spectrogram_190826-103644/epoch-18 --lastfm-path /srv/data/urop/clean_lastfm.db --from-recording -s 30 --cutoff 0.2
```
#### Predict lite

To make prediction simplier, we have produced the `projectname_predict_lite.py` script. We have also uploaded
our checkpoint files under the 'predict' folder so that you may make prediction to your music with our script
without needing to train your own model. The script only relies on the checkpoint and nothing more. Note that
the checkpoint files we uploaded should be the most up-to-date model which gives the best performance. Please
refer to [Requirements](https://github.com/pukkapies/urop2019#requirements) for more details on what you need
to install. 


*Example:*

```
python projectname_predict_lite.py --checkpoint epoch-18 --mp3-path /srv/data/urop/song.mp3
```

Similar to `test_model.py`, you may analyse an entire directory or you may record directly from terminal by
changing the parameters. See `python projectname_predict_lite.py -h` for more details.

## Results

**Experiment 1:**

Below are our results trained from waveform and log mel-spectrogram respectively with the following 50 tags.

Waveform:

Tag used: ['rock', 'female', 'pop', 'alternative', 'male', 'indie', 'electronic', '00s', 'rnb', 'dance', 'hip-hop', 'instrumental', 
'chillout', 'alternative rock', 'jazz', 'metal', 'classic rock', 'indie rock', 'rap', 'soul', 'mellow', '90s', 'electronica', '80s', 
'folk', 'chill', 'funk', 'blues', 'punk', 'hard rock', 'pop rock', '70s', 'ambient', 'experimental', '60s', 'easy listening', 
'rock n roll', 'country', 'electro', 'punk rock', 'indie pop', 'heavy metal', 'classic', 'progressive rock', 'house', 'ballad', 
'psychedelic', 'synthpop', 'trance', 'trip-hop'

The parameters we have used can be found [here](https://github.com/pukkapies/urop2019/blob/master/results/waveform_config_1.json)

![alt text](https://github.com/pukkapies/urop2019/blob/master/results/waveform_1.png)


Log mel-spectrogram:

Tag used: ['rock', 'female', 'pop', 'alternative', 'male', 'indie', 'electronic', '00s', 'rnb', 'dance', 'hip-hop', 'instrumental', 
'chillout', 'alternative rock', 'jazz', 'metal', 'classic rock', 'indie rock', 'rap', 'soul', 'mellow', '90s', 'electronica', '80s', 
'folk', 'chill', 'funk', 'blues', 'punk', 'hard rock', 'pop rock', '70s', 'ambient', 'experimental', '60s', 'easy listening', 
'rock n roll', 'country', 'electro', 'punk rock', 'indie pop', 'heavy metal', 'classic', 'progressive rock', 'house', 'ballad', 
'psychedelic', 'synthpop', 'trance', 'trip-hop']

**Experiment 2:**

Using the same tags and model as the above log-mel-spectrogram experiment above, but this time a cyclical learning rate going between 0.0014/4 and 0.0014 linearly was used. In this experiment a slightly bigger batch size of 128 was also used.

The parameters used can be found [here](https://github.com/pukkapies/urop2019/blob/master/logmelspectrogram_config_1.json)

![alt text](https://github.com/pukkapies/urop2019/blob/master/logmelspectrogram_1.png)

|                                         | AUC-ROC |  AUC-PR |
| --------------------------------------- |:-------:|:-------:|
| Waveform (from us)                      | 86.96   | 39.95   |
| Log mel-spectrogram (from us)           | 87.33   | 40.96   |
| Log mel-spectrogram (cyclic learning rate)           | 87.68   | 42.05   |
| Waveform (Pons, et al., 2018)           | 87.41   | 28.53   |
| Log mel-spectrogram (Pons, et al., 2018)| 88.75   | 31.24   |

In general, we can see that training the MSD dataset on log mel-spectrogram has a better 
performance than training on waveform, which agrees with the result produced by (Pons, et al., 2018).
Note that (Pons, et al., 2018) suggests that when the size of the dataset is large enough, the
quality difference between waveform and log mel-spectrogram model is insignificant (with 1,000,000 songs)

On the other hand, in our experiment, we have cleaned the Last.fm database by
removing tags which are more subjective or have vague meaning, which was not done in (Pons, et al., 2018). According to the results above, the AUC-PR of both waveform and log 
mel-spectrogram has significantly improved from (Pons, et al., 2018) respectively. In the
meantime, the AUC-ROC scores of our experiments are comparable to those produced by
(Pons, et al., 2018). We have therefore shown that training the model using cleaner tags improves the quality of the model. 

## References
Pons, J. et al., 2018. END-TO-END LEARNING FOR MUSIC AUDIO TAGGING AT SCALE. Paris, s.n., pp. 637-644.


## Contacts / Getting Help

calle.sonne18@imperial.ac.uk

chon.ho17@imperial.ac.uk

davide.gallo18@imperial.ac.uk / davide.gallo@pm.me

kevin.webster@imperial.ac.uk
