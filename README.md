# urop2019

Summer UROP 2019 project repository.

## Table of Contents
(make a fancy table of contents)

## Introduction
This project aims to develop a neural network music audio autotagger, i.e. this network will 
take a music audio file and predict a list of tags that are relevant to that audio.

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


## Prerequisites
(hardware and software info that we used)

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
removing unnecessary columns to produce the ‘ultimate’ dataframe. 
This dataframe acts as a clean dataset containing all the essential information 
about the tracks and will be used throughout the project.

For more information about how these functions are used, see ????(the smaller readme)


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
lf = lastfm.LastFm(‘/srv/data/msd/lastfm/SQLITE/lastfm_tags.db’)
```
To use `LastFm2Pandas` (generate dataframe directly from database)

```python
lf = lastfm.LastFm2Pandas(from_sql=‘/srv/data/msd/lastfm/SQLITE/lastfm_tags.db’)
```
To use `LastFm2Pandas` from converted csv,

```python
# generate csv
lastfm.LastFm(‘/srv/data/msd/lastfm/SQLITE/lastfm_tags.db’).db_to_csv(output_dir=’/srv/data/urop’)
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

In 1., 

...We first obtained a list of tags from the Lastfm 
database which have appeared for more than 2000 times. We manually filtered out 
the tags that we considered as non-genre tags and feed genre tags to the algorithm 
`generate_genre_df()`. For each genre tag, the algorithm 
searched for other similar tags from the 500,000 tags 
pool (tags which have occurrence ≥ 10). A new dataset was finally generated with 
the left-column --- the manually chosen tags, the right column 
--- similar matching tags from the pool. 

In 2., 

...We obtained a long list 
of potentially matching tags for each of the four vocal tags. 
We then manually seperate the 'real' matching tags from the rest for each of the lists. 
The lists were fed into `generate_vocal_df()` and a dataset with a similar 
structure as 1) was produced. In the end, the function `generate_final_df()` combined 
the two datasets as a final dataset which was passed to the `lastfm_clean.py`. 

The `.txt` files containing the lists of tags we used in our experiment can be found in 
the folder `~/msd/config`. Hence, if you prefer to use our dataset, you may simply 
generate this by:

```python
generate_final_df(from_csv_path=’/srv/data/urop’, threshold=2000, sub_threshold=10, combine_list=[[‘rhythm and blues’, ‘rnb’], [‘funky’, ‘funk’]], drop_list=[‘2000’, ‘00’, ‘90’, ‘80’, ‘70’, ‘60’])
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



