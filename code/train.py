# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 11:16:10 2018

@author: fj123
@description: Use this script to train the model~=v=~
"""

import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, Callback
import numpy as np
import matplotlib.pyplot as plt
import utils

# define constants
NUMBER_FEATURES_OCTAVE = utils.NUMBER_FEATURES_OCTAVE # 12 midi_notes + sustain + rest + beat_start
NUMBER_FEATURES = utils.NUMBER_FEATURES # 128 midi_notes + sustain + rest + beat_start
INSTRUMENTS = utils.INSTRUMENTS # number of instruments in midifile

# parameters setting =v=
num_steps = 32
batch_size = 256
skip_step = 3
lr = 0.001
dropout = 0.4
decay_rate = 0.1 ** 6
vocabulary = 14
embedding_size = 16
hidden_size = 64
num_epochs = 1000

# paths for training and validation data, experiment path for saving figures and models
train_data_path = 'data\\train'
valid_data_path = 'data\\validation'
experiment_path = 'experiment\\'

# load the training data from the encoded .npy files.
train_data = utils.reload_data_all(train_data_path)
valid_data = utils.reload_data_all(valid_data_path)


# this is the batch generator for generating data during model training.
class KerasBatchGenerator(object):
    
    ###########################################################################
    # data shape:
    #   (number_of_files)(INSTRUMENTS, number_of_ts_in_midi, NUMBER_FEATRUES_OCTAVE)
    # num_steps:
    #   the number of time steps in one unrolled LSTM model
    # batch_size:
    #   number of samples in one mini-batch
    # vocabulary:
    #   number of categories in output
    # skip_step:
    #   steps to skip when generate the training samples
    ###########################################################################
    
    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=3):
        # save the variables, indexes
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.current_index = 0 # index of t in current midifile
        self.file_index = 0 # the index of current midifile
        self.skip_step = skip_step
        
        
    # generate function called at the beginning of each batch, yield input and output data
    def generate(self):
        # define dimensions for inputs and outputs
        x = np.zeros((self.batch_size, self.num_steps, INSTRUMENTS * NUMBER_FEATURES_OCTAVE), dtype=np.bool)
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary), dtype=np.bool)
        
        while True:
            for i in range(self.batch_size):
                # the ith sample in the current batch
                
                if self.current_index + self.num_steps >= len(self.data[self.file_index][0]):
                    self.current_index = 0
                    self.file_index = (self.file_index + 1) % len(self.data)
                    
                while len(self.data[self.file_index][0]) < self.num_steps:
                    self.file_index = (self.file_index + 1) % len(self.data)
                    
                # set values for input and output data
                x[i, :, :NUMBER_FEATURES_OCTAVE] = self.data[self.file_index][0, self.current_index : self.current_index + self.num_steps, :]
                x[i, 1:, NUMBER_FEATURES_OCTAVE:] = self.data[self.file_index][1, self.current_index : self.current_index + self.num_steps - 1, :]
                y[i, :, :] = self.data[self.file_index][1, self.current_index : self.current_index + self.num_steps, :self.vocabulary]
                
                self.current_index += self.skip_step
                
            yield x, y


# train and validation data batch generators.
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size,
                                           vocabulary, skip_step=skip_step)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size,
                                           vocabulary, skip_step=skip_step)


# define model structure
model = Sequential()
model.add(Dropout(dropout, input_shape=(num_steps, NUMBER_FEATURES_OCTAVE * INSTRUMENTS)))
model.add(TimeDistributed(Dense(embedding_size)))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(dropout))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))
optimizer = Adam(lr=lr, decay=decay_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])
print(model.summary())

# save model at end of each epoch
checkpointer = ModelCheckpoint(filepath=experiment_path + 'model{epoch:02d}-valloss{val_loss:.2f}-valacc{val_acc:.2f}.hdf5', verbose=1)

# plot losses at the end of each epoch
class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.logs = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        plt.figure()
        plt.plot(self.x, self.losses, 'r--', label='loss')
        plt.plot(self.x, self.val_losses, 'g--', label='val_loss')
        plt.title('loss vs. valid_loss on epoch{}'.format(self.i))
        plt.legend()
        plt.grid()
        plt.savefig(experiment_path + 'loss vs.valid_loss epoch{}.svg'.format(self.i), format='svg')
        plt.show()
        
plot_losses = PlotLosses()

# calculate the number of steps needed for each epoch (one step per batch)
steps_per_epoch = 0
for i in range(len(train_data)):
    steps_per_epoch += len(train_data[i][0] - num_steps) // (batch_size * skip_step)
validation_steps = 0
for i in range(len(valid_data)):
    validation_steps += len(valid_data[i][0] - num_steps) // (batch_size * skip_step)


# model training!
His = model.fit_generator(train_data_generator.generate(), steps_per_epoch, num_epochs,
                    validation_data=valid_data_generator.generate(),
                    validation_steps=validation_steps,
                    callbacks=[checkpointer, plot_losses])

# plot losses in the end.
plt.figure()
plt.plot(His.history['loss'], 'r--', label='loss')
plt.plot(His.history['val_loss'], 'g--', label='validation loss')
plt.title('final loss vs. valid_loss')
plt.legend()
plt.grid(linestyle = "--")
plt.savefig(experiment_path + 'final loss vs. valid_loss.svg', format='svg')
plt.show()

# plot accuracies in the end~
plt.figure()
plt.plot(His.history['acc'], 'r--', label='accuracy')
plt.plot(His.history['val_acc'], 'g--', label='validation accuracy')
plt.title('final accuracy vs. valid_accuracy')
plt.legend()
plt.grid(linestyle = "--")
plt.savefig(experiment_path + 'final accuracy vs. valid_accuracy.svg', format='svg')
plt.show()

# save losses and accuracies during training~
np.save(experiment_path + 'History.loss.npy', His.history['loss'])
np.save(experiment_path + 'History.val_loss.npy', His.history['val_loss'])
np.save(experiment_path + 'History.acc.npy', His.history['acc'])
np.save(experiment_path + 'History.val_acc.npy', His.history['val_acc'])

# save the final model at the last epoch
model.save(experiment_path + 'final_model.hdf5')


                    

















