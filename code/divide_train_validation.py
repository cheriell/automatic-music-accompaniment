# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 13:12:45 2018

@author: fj123
@description: This script will copy the encoded data representations in the dataset into 
data\\train and data\\validation folders. The dividing portion for train and validation 
will be 4:1.
"""

import os
import utils
import numpy as np

# get all the datapaths in the dataset
path = '..\\midis\\'
try:
    data_paths = [os.path.join(path, o) \
                  for o in os.listdir(path) \
                  if os.path.isdir(os.path.join(path, o))]
except OSError as e:
    print('Error: Invalid datapath!!!')

count = 0

# copy the data files in to data\\train and data\\validation
# the file names are changed to be index numbers of the music pieces.
for data_path in data_paths:
    midi_datas = utils.get_data_paths(data_path)
    
    for midi_data in midi_datas:

    	data_cur = np.load(midi_data)
    	if count % 5 == 4:
    		np.save('data\\validation\\{}.npy'.format(count), data_cur)
    	else:
    		np.save('data\\train\\{}.npy'.format(count), data_cur)

    	count += 1

    print(data_path + ' done!')
