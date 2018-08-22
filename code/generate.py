# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 21:31:19 2018

@author: fj123
@description: The script will use a selected model to generate accompaniment for midi melodies.
use command line arguments:
    midifile.mid --model_file model.hdf5 --diversity divf
the midiefile is the midi that includes the melody you want to generate accomapniemnt with, model_file
should be a model that have an input dimension of (batch_size, num_steps, INSTRUMENTS * NUMBER_FEATRUES_OCTAVE) and 
an output dimension of (batch_size, num_steps, vacabulary).
"""


from keras.models import load_model
import numpy as np
import utils
import argparse

# define constants
NUMBER_FEATURES_OCTAVE = utils.NUMBER_FEATURES_OCTAVE # 12 midi_notes + sustain + rest + beat_start
NUMBER_FEATURES = utils.NUMBER_FEATURES # 128 midi_notes + sustain + rest + beat_start
INSTRUMENTS = utils.INSTRUMENTS # number of instruments in midifile
num_steps = 32
vocabulary = 14

# add parser arguments and pass the received arguments into variables.
parser = argparse.ArgumentParser()
parser.add_argument('midi_file', type=str, default=None, help='The midi file for melody input')
parser.add_argument('--model_file', type=str, default='final_model.hdf5', help='model file for accompaniment generation')
parser.add_argument('--diversity', type=float, default=0.6, help='diversity in accompaniment generation')
args = parser.parse_args()

midi_file = args.midi_file
model_file = args.model_file
diversity = args.diversity

# load the deep learning model and data for the melodeis
model = load_model(model_file)
test_data_raw = utils.load_melody_data(midi_file)
utils.to_monophonic(test_data_raw)
test_data = utils.to_octave(test_data_raw)


# sample a note from the probability distribution.
# This helper function is copied from keras lstm examples at:
# https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(prediction, diversity=0.6):
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / diversity
    prediction_exp = np.exp(prediction)
    prediction = prediction_exp / np.sum(prediction_exp)
    probs = np.random.multinomial(1, prediction, 1)
    return np.argmax(probs)


# generate accompaniment from left to right.
test_data[1, num_steps:, :] = 0
i = 0
while i + num_steps < len(test_data[0]):
    x = np.zeros((1, num_steps, INSTRUMENTS * NUMBER_FEATURES_OCTAVE), dtype=np.bool)
    x[:, :, :NUMBER_FEATURES_OCTAVE] = test_data[0, i : i + num_steps, :]
    x[:, 1:, NUMBER_FEATURES_OCTAVE:] = test_data[1, i : i + num_steps - 1, :]
    
    prediction = model.predict(x)
    # predict_note = np.argmax(prediction[0, num_steps - 1, :])
    predict_note = sample(prediction[0, num_steps - 1, :], diversity)
    
    test_data[1, i + num_steps - 1, predict_note] = 1
    
    i += 1
    
data = utils.reverse_octave(test_data)

# combine the generated accompaniment with the original melodeis.
data_new = np.copy(test_data_raw)
data_new[1, :, :] = data[1, :, :]

# print the generated new music piece and save the music into midi file.
utils.print_data(data_new)
utils.generate_midi(data_new, midi_file[:len(midi_file)-4] + '-generate-' + model_file[:len(model_file)-5] + '-{}.mid'.format(diversity))

