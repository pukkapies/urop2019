This folder contains the script needed to "properly" clean the Million Songs Dataset. We implicitly assume you _do_ have access to the dataset (how to get it? really good question...). We also assume you downloaded the Last.fm database from [here](http://millionsongdataset.com/lastfm/), or that you manually obtained it using the Last.fm API. 

## Database and Audio Cleaning

Each of the following scripts can be run in terminal by typing `python <script name.py>`. Each script has an `--help` page containing all the information you might need about options and parameters. Here is the correct workflow:

 - use `python fetcher.py <options> output.csv` to create a .csv file containing a list of file paths, the 7Digital ID of the tracks, and some extra information from the Million Songs Dataset tracks saved locally (the filenames are supposed to be of the form 7Digital ID + '.clip.mp3');

- use `python wrangler.py <options> input.csv output.csv` to read the .csv file generated by fetcher.py and create a .csv file linking each track to its track ID (the HDF5 summary file can be download [here](http://millionsongdataset.com/sites/default/files/AdditionalFiles/msd_summary_file.h5)), and also to remove unwanted entries such as mismatches, duplicates or files which can't be opened (read [here](http://millionsongdataset.com/blog/12-2-12-fixing-matching-errors/) about the mismatches issue; read [here](http://millionsongdataset.com/blog/11-3-15-921810-song-dataset-duplicates/) about duplicates in the dataset);

- use `python mp3_to_npz.py <options> input.csv` to read the .csv file generated by wrangler.py (or by fetcher.py, if you also want mismatched tracks) and create the .npz files which will allow us to spot silent tracks and analyse silent sections within non-silent tracks;

- use `python wrangler_info_silence.py <options> input.csv output.csv` to read the .csv file generated by wrangler.py (or, again, by fetcher.py) and look for the .npz files generated by mp3_to_npz.py in order to create a file containing _only_ the tracks which satisfy certain requirements, such as a minimal "effective length" or a maximal amount of silence (that is, the final tracks that you want to use in your model);

- use `python lastfm_cleaning.py` to read the final .csv file generated by wrangler_info_silence.py and produce a final tags database (similar in structure to the original one, in order to allow compatibility with our querying module) containing only clean and meaningful tags (lastfm_cleaning_tools.py does the heavy lifting here, if you are planning to check out the code).

You will now have a .csv file containing only the tracks that you are happy to use, and a new .db file containing tags information for each (clean) track.

## Tags Cleaning
Below is a brief tutorial of how to customise the tags filtering. For more details, please see the documentaion of `lastfm_cleaning_utils.py`

Step 1: Produce the popularity dataset. This dataset contains ranking of tags based on the number of occurrences in the lastfm database. 

```python
from lastfm import LastFm2Pandas
popularity = LastFm2Pandas.from_csv('/srv/data/urop').popularity()
```

Step 2: Generate five txt files (non_genre_list.txt, male_list.txt, female_list.txt, vocal_list.txt, instrumental_list.txt)

```python
generate_genre_txt(popularity, threshold=2000)
generate_vocal_txt(df: pd.DataFrame, tag_list = ['rap', 'instrumental', 'male', 'female'], percentage_list=[90, 90, 90, 80])
```

Use ` set_txt_path()` to set the default path to store the files.

Step 3: Open the text files and put a symbol ‘-’ in front of the tags that you would like to filter out, and rename the amended .txt files name by putting a suffix ‘_filtered’. For example: ‘male_list.txt’ should be renamed to ‘male_list_filtered.txt’.

Step 4: Feed the `.txt` files into the algorithm

```python
generate_genre_df(popularity, threshold=2000, sub_threshold = 10)
generate_vocal_df(indicator='-')
```
where sub_threshold=10 indicates that the algorithm will only run a search on tags in the 500,000 tag pool which have occurrence of at least 10.

Step 5: Take a look at the generated datasets from step 4 and decide what further tags you may want to add, merge, or remove from the datasets. After that, run the function generate_final_df() with your chosen parameters. For example,

```python
generate_final_df(from_csv_path=’/srv/data/urop’, threshold=2000, sub_threshold=10, combine_list=[[‘rhythm and blues’, ‘rnb’], [‘funky’, ‘funk’]], drop_list=[‘2000’, ‘00’, ‘90’, ‘80’, ‘70’, ‘60’])
```

Alternatively, you may skip step 4 but directly run step 5. After taking a look at the output dataset, you may use the functions `combine_tags()`, `add_tags()`, `remove()` to customise your datasets.

Please refer to the documentation of lastfm_cleaning_utils.py for more details.