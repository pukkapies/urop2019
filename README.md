# urop2019 -- Projectname

Summer UROP 2019 project repository.

This project aims to develop a neural network music audio autotagger, i.e. this network will 
take a music audio file and predict a list of tags that are relevant to that audio.

## Table of Contents


* [Introduction](https://github.com/pukkapies/urop2019#introduction)
* [Prerequisites](https://github.com/pukkapies/urop2019#prerequisites)
* [Data Cleaning](https://github.com/pukkapies/urop2019#data-cleaning)
    * [Audio](https://github.com/pukkapies/urop2019#audio)
    * [Database](https://github.com/pukkapies/urop2019#database)
    * [Tags](https://github.com/pukkapies/urop2019#tags)
* [Data Input Pipeline](https://github.com/pukkapies/urop2019#data-input-pipeline)
    * [TFRecords](https://github.com/pukkapies/urop2019#tfrecords)
    * [Dataset Preparation](https://github.com/pukkapies/urop2019#dataset-preparation)
* [Training](https://github.com/pukkapies/urop2019#training)
    * [Model and Configuration](https://github.com/pukkapies/urop2019#model-and-configuration)
    * [Training loops](https://github.com/pukkapies/urop2019#training-loops)
* [Evaluation tools](https://github.com/pukkapies/urop2019#evaluation-tools)
* [Results](https://github.com/pukkapies/urop2019#results)
* [Reference](https://github.com/pukkapies/urop2019#reference)
* [Contact / Getting Help](https://github.com/pukkapies/urop2019#contact--getting-help)


## Introduction

This project makes use of the freely-available [Million Song Dataset]( http://millionsongdataset.com), 
and the [Last.fm](http://millionsongdataset.com/lastfm/) dataset. The former provides a link between 
all the useful information of the related to the tracks and the actual track files, whereas the 
latter contains all the tags information of the audio files.

Outline of the project:

1. Extract, clean, and merge all the useful information from the Million Song Dataset 
and the Last.fm dataset to produce final datasets that will be used in training.

2. Prepare data input pipelines and transform the data as tf.data.Dataset that 
will be consumed by the training algorithm.

3. Create flexible training algorithms and tools for model evaluation that allow 
customised experiments.

4.  Train a neural network and produce an algorithm that is capable of making sensible 
genre predictions to input audio.

In the following sections, we will present to you in more details on what we have done, 
and a brief tutorial of how you may our codes to make genre predictions to your piece of 
audio, or even how to make use of our codes to carry out your experiment very easily.

If you are interested to see our experiment result, click here (hyperlink to results)


## Prerequisites

* Python 3.6
* One or more CUDA-enabled GPUs
* Mac or Linux environment
* Libraries Required:
    * [TensorFlow](https://www.tensorflow.org/beta) 2.0.0-rc0 GPU version or above
    * [Librosa](https://github.com/librosa/librosa) 0.7.0 + ffmpeg
    * [Mutagen](https://mutagen.readthedocs.io/en/latest/) 1.42.0
    * Other common libraries 

## Data Cleaning

### Audio
Firstly, by `fetcher.py`, the directory which contains all the tracks is thoroughly 
scanned. The info: file path, duration,  number of channels, file size are captured and 
stored in a Pandas dataframe. The audio files that cannot be opened correctly are removed 
from the dataframe.  

After that, `mp3_to_numpy.py` uses the librosa library to convert every audio file that can 
be opened into numpy arrays (based on the number of channels). The numpy arrays of each track 
are then analysed to extract the location of any silent sections respectively (see the 
documentation in the script for more details). The silent information, the arrays, and the 
sampling rate of each track are optionally stored as an npz file in the given directory. 

The silent information is processed and interpreted by `wrangler_silence.py`, 
and the results, e.g. effective_clip_length, max_silence_length are appended to 
the dataframe. `wrangler_silence` provides functions that can filter out tracks 
based on their silent information. In our experient, tracks with ??????????????????? 


**Example**:

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
python wrangler_silence.py /srv/data/urop2019/fetch.csv /srv/data/urop2019/wrangle_silence.csv --root-dir-npz /srv/data/urop2019/npz --root-dir-mp3 /srv/data/msd/7digital --min-size ? --filter-tot-silence 15 --filter-max-silence ?
```

### Database 
The raw HDF5 Million Song Dataset file, which contains three smaller datasets, 
are converted into multiple Pandas dataframes. The relevant information is then 
extracted and merged. According to the MSD website, there are mismatches between 
these datasets. For more details, see [here](http://millionsongdataset.com/blog/12-2-12-fixing-matching-errors/). 
To deal with this issue, `wrangler.py` takes a '.txt' file with a list 
of tids which could not be trusted, and remove the corresponding rows 
of data based on the list. Furthermore, MSD also provides `.txt` file 
with a list of tracks that have duplicates. `wrangler.py` by default 
keeps one version of the duplicate tracks of each song according to the list 
and remove the rest. 

The dataframe from above is merged with the dataframe 
produced by the audio section above followed by 
removing unnecessary columns to produce the 'ultimate' dataframe. 
This dataframe acts as a clean dataset containing all the essential information 
about the tracks and will be used throughout the project.


For more information about how these functions are used, see [here](https://github.com/pukkapies/urop2019/blob/master/msd/README.md)



**Example**

```
python wrangle.py /srv/data/urop2019/wrangle_silence.csv /srv/data/urop2019/ultimate.csv --path-h5 /srv/data/msd/entp/msd_summary_file.h5 --path-db /srv/data/msd/lastfm/SQLITE/lastfm_tags.db --path-txt-dupl ???? --path-txt-mism /srv/data/msd/sid_mismatches.txt
```

Alternatively, to save storage space and time, the following order of code 
execution was used instead:

```
python fetch.py /srv/data/urop2019/fetch.csv --root-dir /srv/data/msd/7digital
```

```
python wrangle.py /srv/data/urop2019/fetch.csv /srv/data/urop2019/fetch2.csv --path-h5 /srv/data/msd/entp/msd_summary_file.h5 --path-db /srv/data/msd/lastfm/SQLITE/lastfm_tags.db --path-txt-dupl ???? --path-txt-mism /srv/data/msd/sid_mismatches.txt --discard-dupl False
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

### Tags
#### Make Queries
`lastfm.py` contains two classes, `LastFm`, `LastFm2Pandas`, that each of them contains 
all the basic tools for querying the Lastfm database. The former directly queries 
the database by SQL, whereas the latter converts the database into csv files and 
queries the data using the Pandas library. In some of the functions in latter sections, 
it may have a `lastfm` input parameter and require to be set as an instance of one
of the classes. 

**Example:**

To use `LastFm`,

```python
lf = lastfm.LastFm('/srv/data/msd/lastfm/SQLITE/lastfm_tags.db')
```
To use `LastFm2Pandas` (generate dataframe directly from database)

```python
lf = lastfm.LastFm2Pandas(from_sql='/srv/data/msd/lastfm/SQLITE/lastfm_tags.db')
```
To use `LastFm2Pandas` from converted csv,

```python
# generate csv
lastfm.LastFm('/srv/data/msd/lastfm/SQLITE/lastfm_tags.db').db_to_csv(output_dir='/srv/data/urop')
# create class instance
lf = lastfm.LastFm2Pandas(from_csv='/srv/data/urop')
```

Note that the major difference between the two classes is that 
`LastFm` is quicker to initiate, but some queries might take some time to 
perform, whereas `LastFm2Pandas` may take longer to initiate due to the whole 
dataset being loaded to the memory. However, it contains some more advanced methods, and
it is quick to initialise if database is converted into csv files in advance.


To explore the Million Song Dataset summary file, `metadata.py` contains 
basic tools to query the `msd_summary_file.h5` file. 

#### Filtering

In the Lastfm database, there are more than 500,000 different tags. 
To ensure that the training algorithm can learn from more sensible tags, 
the tags are cleaned using `lastfm_cleaning_utils.py`. The exact mechanisms 
of how it works can be found in the documentation of the script. 
In brief, the tags are divided into two categories: 

1. genre tags 

2. vocal tags (male, female, rap, instrumental) 

In our experiment: 

In 1. 

* We first obtained a list of tags from the Lastfm 
database which have appeared for more than 2000 times. We manually filtered out 
the tags that we considered as non-genre tags and fed the genre tags to the algorithm 
`generate_genre_df()`. For each genre tag, the algorithm 
searched for other similar tags from the 500,000 tags 
pool (tags which have occurrence ≥ 10). A new dataset was finally generated with 
the left-column --- the manually chosen tags, the right column 
--- similar matching tags from the pool. 

In 2. 

* We obtained a long list of potentially matching tags for each of the four vocal tags. 
We then manually seperate the 'real' matching tags from the rest for each of the lists. 
The lists were fed into `generate_vocal_df()` and a dataset with a similar structure as 
1. was produced. In the end, the function `generate_final_df()` combined the two 
datasets as a final dataset which was passed to the `lastfm_clean.py`. 

The `.txt` files containing the lists of tags we used in our experiment can be found in 
the folder `~/msd/config`. Hence, if you prefer to use our dataset, you may simply 
generate this by:

```python
generate_final_df(from_csv_path='/srv/data/urop', threshold=2000, sub_threshold=10, combine_list=[['rhythm and blues', 'rnb'], ['funky', 'funk']], drop_list=['2000', '00', '90', '80', '70', '60'])
```

if you are interested to view the dataset. Otherwise, `lastfm_clean.py` will automatically 
generate this dataset and transform it into a clean Lastfm database. 

Note that `lastfm_cleaning_utils` allows a great deal of customisation. 
Please see (small readme) for more details.

`lastfm_cleaning.py` creates a new database file using the cleaned tags 
from lastfm_cleaning_utils.py. The database has the same structure as the 
`lastfm_tags.db` database, and can be queried by `lastfm.py`.

**Example:**

```
python lastfm_cleaning.py /srv/data/msd/lastfm/SQLITE/lastfm_tags.db /srv/data/urop/clean_lastfm.db --val ?? 
```

To summeraise, `metadata.py` was used to convert between TID and 7digitalid as the `.mp3` 
files were named by 7digitalid whilst tag information was linked to the TID. 
THIS SCRIPT WAS NEVER USED??

`lastfm.py` was used throughout the project whenever tag information is needed to be fetched. 
In our experiment, `lastfm_cleaning` and `lastfm_cleaning_utils` were used once together 
in order to generate the `clean_lastfm.db` containing 155 clean tags. We stored 
more tags than we would probably need, but this was better than potentially 
having to regenerate the TFRecord files.


## Data Input Pipeline
### Tfrecords
To store the necessary information we need in training, we used the `.tfrecord` 
file format. The `preprocessing.py` script does exactly this. In each entry of 
the `.tfrecord` file, it stores the audio as an array in the formats of either 
waveform or log mel-spectrogram. It will also store the TID to identify each 
track as well as the tags from the `clean_lastfm.db` database in a one-hot vector format.

`preprocessing.py` leaves quite a lot of room for user customisation. It will 
accept audio as `.mp3` files, or as `.npz` files where each entry contains the 
audio as an array and the sample rate. The user can choose the sample rate to 
store the data in as well as the number of mel bins when storing the audios in 
the log mel-spectrogram form. It is also possible to specify the number of  
`.tfrecord` files to split the data between.

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

### Dataset Preparation
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

**Example:**

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

## Training
### Model and Configuration


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

**Example:**

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

### Training Loops

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

**Example**
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


## Evaluation Tools
`test_model.py` is the script containing the evaluation tools. There is a `test_model()` function which 
simply tests the model's performance on the test dataset from a certain checkpoint. The `predict()` function 
takes an audio array, in the waveform or the log mel-spectrogram format, and uses the model on consecutive 15s 
windows of the input audio to returns the average prediction as tags in string form.

(example?)

## Results

**Experiment 1:**

Below are our results trained from waveform and log mel-spectrogram respectively with the following 50 tags.

Waveform:

Tag used: ['rock', 'female', 'pop', 'alternative', 'male', 'indie', 'electronic', '00s', 'rnb', 'dance', 'hip-hop', 'instrumental', 'chillout',
 'alternative rock', 'jazz', 'metal', 'classic rock', 'indie rock', 'rap', 'soul', 'mellow', '90s', 'electronica', '80s', 'folk', 'chill',
 'funk', 'blues', 'punk', 'hard rock', 'pop rock', '70s', 'ambient', 'experimental', '60s', 'easy listening', 'rock n roll', 'country',
 'electro', 'punk rock', 'indie pop', 'heavy metal', 'classic', 'progressive rock', 'house', 'ballad', 'psychedelic', 'synthpop',
 'trance', 'trip-hop']

The parameters we have used can be found [here](https://github.com/pukkapies/urop2019/blob/master/results/waveform_config_1.json)

![alt text](https://github.com/pukkapies/urop2019/blob/master/results/waveform_1.png)


Log mel-spectrogram:

Tag used: ['rock', 'female', 'pop', 'alternative', 'male', 'indie', 'electronic', '00s', 'rnb', 'dance', 'hip-hop', 'instrumental', 'chillout',
 'alternative rock', 'jazz', 'metal', 'classic rock', 'indie rock', 'rap', 'soul', 'mellow', '90s', 'electronica', '80s', 'folk', 'chill',
 'funk', 'blues', 'punk', 'hard rock', 'pop rock', '70s', 'ambient', 'experimental', '60s', 'easy listening', 'rock n roll', 'country',
 'electro', 'punk rock', 'indie pop', 'heavy metal', 'classic', 'progressive rock', 'house', 'ballad', 'psychedelic', 'synthpop',
 'trance', 'trip-hop']


The parameters we have used can be found [here](https://github.com/pukkapies/urop2019/blob/master/results/logmelspectrogram_config_1.json)

![alt text](https://github.com/pukkapies/urop2019/blob/master/results/logmelspectrogram_1.png)

|                                         | AUC-ROC |  AUC-PR |
| --------------------------------------- |:-------:|:-------:|
| Waveform (from us)                      |         |         |
| Log mel-spectrogram (from us)           |         |         |
| Waveform (Pons, et al., 2018)           | 87.41   | 28.53   |
| Log mel-spectrogram (Pons, et al., 2018)| 88.75   | 31.24   |

In general, we can see that training the MSD dataset on log mel-spectrogram has a better 
performance than training on waveform, which agrees with the result produced by (Pons, et al., 2018).
Note that (Pons, et al., 2018) suggests that when the size of the dataset is large enough, the
quality difference between waveform and log mel-spectrogram model is insignificant (with 1,000,000 songs)

On the other hand, in our experiment, we have cleaned the Last.fm database by
removing tags which are more subjective or have vague meaning, which was not done in (Pons, et al., 2018). 
According to the results above, the AUC-PR of both waveform and log 
mel-spectrogram has significantly improved from (Pons, et al., 2018) respectively. In the
meantime, the AUC-ROC scores of our experiments are comparable to those produced by
(Pons, et al., 2018). We have, therefore, proven that training
the model on cleaner tags with both waveform and log mel-spectrogram will improve the quality
of the models. 

(any further analysis?)

## Reference
Pons, J. et al., 2018. END-TO-END LEARNING FOR MUSIC AUDIO TAGGING AT SCALE. Paris, s.n., pp. 637-644.


## Contact / Getting Help

calle.sonne18@imperial.ac.uk

chon.ho17@imperial.ac.uk

davide.gallo18@imperial.ac.uk








