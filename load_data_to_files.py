# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 13:12:45 2018

@author: fj123
"""

import utils
import numpy as np
import os


path = '..\\midis\\'
try:
    data_paths = [os.path.join(path, o) \
                  for o in os.listdir(path) \
                  if os.path.isdir(os.path.join(path, o))]
except OSError as e:
    print('Error: Invalid datapath!!!')


for data_path in data_paths:
    midi_files = utils.get_file_paths(data_path)
    
    for midi_file in midi_files:
        data_cur = utils.load_data(midi_file)
        utils.to_monophonic(data_cur)
        data_cur = utils.to_octave(data_cur)
        
        datafile = midi_file[:len(midi_file) - 4] + '.npy'
        np.save(datafile, data_cur)

