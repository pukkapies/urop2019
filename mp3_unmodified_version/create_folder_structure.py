# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:21:24 2019

@author: MacBook Pro
"""

import os

inputpath = '~/srv/data/msd/7digital/'
outputpath = '~/srv/data/urop/7digital_numpy/'

for dirpath, dirnames, filenames in os.walk(inputpath):
    structure = os.path.join(outputpath, dirpath[len(inputpath):])
    if not os.path.isdir(structure):
        os.mkdir(structure)
    else:
        print("Folder does already exits!")
        
