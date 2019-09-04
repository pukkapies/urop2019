This folder contains the script needed to "properly" clean the Million Songs Dataset. We implicitly assume you _do_ have access to the dataset (how to get it? really good question...). We also assume you downloaded the Last.fm database from [here](http://millionsongdataset.com/lastfm/), or that you manually obtained it using the Last.fm API. 

Each of the following scripts can be run in terminal by typing `python <script name.py>`. Each script has an `--help` page containing all the information you might need about options and parameters. Here is the correct workflow:

 - use `python fetcher.py <options> output.csv` to create a .csv file containing a list of file paths, the 7Digital ID of the tracks, and some extra information from the Million Songs Dataset tracks saved locally (the filenames are supposed to be of the form 7Digital ID + '.clip.mp3');

- use `python wrangler.py <options> input.csv output.csv` to read the .csv file generated by fetcher.py and create a .csv file linking each track to its track ID (the HDF5 summary file can be download [here](http://millionsongdataset.com/sites/default/files/AdditionalFiles/msd_summary_file.h5)), and also to remove unwanted entries such as mismatches, duplicates or files which can't be opened (read [here](http://millionsongdataset.com/blog/12-2-12-fixing-matching-errors/) about the mismatches issue; read [here](http://millionsongdataset.com/blog/11-3-15-921810-song-dataset-duplicates/) about duplicates in the dataset);

- use `python mp3_to_npz.py <options> input.csv` to read the .csv file generated by wrangler.py (or by fetcher.py, if you also want mismatched tracks) and create the .npz files which will allow us to spot silent tracks and analyse silent sections within non-silent tracks;

- use `python wrangler_info_silence.py <options> input.csv output.csv` to read the .csv file generated by wrangler.py (or, again, by fetcher.py) and look for the .npz files generated by mp3_to_npz.py in order to create a file containing _only_ the tracks which satisfy certain requirements, such as a minimal "effective length" or a maximal amount of silence (that is, the final tracks that you want to use in your model);

- use `python lastfm_cleaning.py` to read the final .csv file generated by wrangler_info_silence.py and produce a final tags database (similar in structure to the original one, in order to allow compatibility with our querying module) containing only clean and meaningful tags (lastfm_cleaning_tools.py does the heavy lifting here, if you are planning to check out the code).

You will now have a .csv file containing only the tracks that you are happy to use, and a new .db file containing tags information for each (clean) track.