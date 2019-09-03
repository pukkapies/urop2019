# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:28:58 2019

@author: hcw10
"""

import argparse
import sounddevice as sd
import time

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="path or filename.wav or filename")
parser.add_argument("-s", "--second", help="number of seconds to record", type=int, default='10')

args = parser.parse_args()

from scipy.io.wavfile import write

fs = 16000  # Sample rate
seconds = int(args.second)  # Duration of recording

if args.filename[:-4] != '.wav':
    args.filename = args.filename + '.wav'


while True:
    val = input('Press Enter to begin')
    if val is not None:
        break

print('record starts in')
print('3')
time.sleep(1)
print('2')
time.sleep(1)
print('1')
time.sleep(1)
print('0')
print('Recording')

numpy_record = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
print(numpy_record[:10])
write(args.filename, fs, numpy_record)  # Save as WAV file

print('Finished recording')

# Save the recorded data as a WAV file


