To make prediction simplier, we have produced the `orpheus_lite.py` script. We have also uploaded our checkpoint file containing the trained weights so that you may make prediction to your music without needing to train your own model. The script only relies on the checkpoint and nothing more. The checkpoint files we uploaded should be the most up-to-date, which would give the best performance.


*Example:*

```
python orpheus_lite.py --checkpoint /path/to/checkpoint --mp3 /path/to/your/song.mp3
```

Similarly to the full `orpheus.py` script, you may analyse an entire audio directory, or you may record audio directly from terminal. See `python orpheus_lite.py -h` for more details on the correct command line syntax.
