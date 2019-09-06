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
Firstly, by `track_fetch.py`, the directory which contains all the tracks is thoroughly 
scanned. The info: file path, duration,  number of channels, file size are captured and 
stored in a Pandas dataframe. The audio files that cannot be opened correctly are removed 
from the dataframe.  

After that, `mp3_to_numpy.py` uses the librosa library to convert every audio file that can 
be opened into numpy arrays (based on the number of channels). The numpy arrays of each track 
are then analysed to extract the location of any silent sections respectively (see the 
documentation in the script for more details). The silent information, the arrays, and the 
sampling rate of each track are optionally stored as an npz file in the given directory. 

The silent information is processed and interpreted by `track_wrangle_silence.py`, 
and the results, e.g. effective_clip_length, max_silence_length are appended to 
the dataframe. `track_wrangle_silence` provides functions that can filter out tracks 
based on their silent information. In our experient, tracks with ??????????????????? 


**Example**:

This will create a .csv that contains the information of tracks mentioned above.

`python fetch.py /srv/data/urop2019/fetch.csv --root-dir /srv/data/msd/7digital`

This will generate npz files mentioned above.

```python mp3_to_numpy.py /srv/data/urop2019/fetch.csv --root-dir-npz /srv/data/urop2019/npz --root-dir-mp3 
/srv/data/msd/7digital```

This will expand the fetch.csv generated above to include some interpretations 
of the silent information.

```python wrangler_silence.py /srv/data/urop2019/fetch.csv 
/srv/data/urop2019/wrangle_silence.csv --root-dir-npz 
/srv/data/urop2019/npz --root-dir-mp3 /srv/data/msd/7digital 
--min-size ? --filter-tot-silence 15 --filter-max-silence ?```

### Database 
The raw HDF5 Million Song Dataset file, which contains three smaller datasets, 
are converted into multiple Pandas dataframes. The relevant information is then 
extracted and merged. According to the MSD website, there are mismatches between 
these datasets. For more details, see [here](http://millionsongdataset.com/blog/12-2-12-fixing-matching-errors/). 
To deal with this issue, track_wrangle.py takes a '.txt' file with a list 
of tids which could not be trusted, and remove the corresponding rows 
of data based on the list. Furthermore, MSD also provides `.txt` file 
with a list of tracks that have duplicates. `track_wrangle.py` by default 
keeps one version of the duplicate tracks of each song according to the list 
and remove the rest. 

The dataframe from above is merged with the dataframe 
produced by the audio section above followed by 
removing unnecessary columns to produce the ‘ultimate’ dataframe. 
This dataframe acts as a clean dataset containing all the essential information 
about the tracks and will be used throughout the project.

For more information about how these functions are used, see ????(the smaller readme)


**Example**

```python wrangle.py /srv/data/urop2019/wrangle_silence.csv /srv/data/urop2019/ultimate.csv 
--path-h5 /srv/data/msd/entp/msd_summary_file.h5 --path-db 
/srv/data/msd/lastfm/SQLITE/lastfm_tags.db --path-txt-dupl ???? 
--path-txt-mism /srv/data/msd/sid_mismatches.txt```

Alternatively, to save storage space and time, the following order of code 
execution was used instead:

```python fetch.py /srv/data/urop2019/fetch.csv --root-dir /srv/data/msd/7digital```

```python wrangle.py /srv/data/urop2019/fetch.csv /srv/data/urop2019/fetch2.csv 
--path-h5 /srv/data/msd/entp/msd_summary_file.h5 --path-db 
/srv/data/msd/lastfm/SQLITE/lastfm_tags.db --path-txt-dupl ???? 
--path-txt-mism /srv/data/msd/sid_mismatches.txt --discard-dupl False```

```python mp3_to_numpy.py /srv/data/urop2019/fetch2.csv 
--root-dir-npz /srv/data/urop2019/npz --root-dir-mp3 
/srv/data/msd/7digital```

```python wrangler_silence.py /srv/data/urop2019/fetch2.csv 
/srv/data/urop2019/wrangle_silence.csv --root-dir-npz /srv/data/urop2019/npz 
--root-dir-mp3 /srv/data/msd/7digital --min-size ? --filter-tot-silence 15 
--filter-max-silence ?```

```python wrangle.py /srv/data/urop2019/wrangle_silence.csv 
/srv/data/urop2019/ultimate.csv --path-h5 /srv/data/msd/entp/msd_summary_file.h5 
--path-db /srv/data/msd/lastfm/SQLITE/lastfm_tags.db --path-txt-dupl ???? 
--path-txt-mism /srv/data/msd/sid_mismatches.txt```

With this order of execution, `wrangle.py` will remove tracks which 
have no tags. This reduces the number of tracks from 1,000,000 to 500,000+.

For more information on how you can customise the procedures, 
see the documentation in the corresponding scripts.
