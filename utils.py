# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 11:19:13 2018

@author: fj123
"""

import numpy as np
import os, sys
import pretty_midi

NUMBER_FEATURES_OCTAVE = 15 # 12 midi_notes + sustain + rest + beat_start
NUMBER_FEATURES = 131 # 128 midi_notes + sustain + rest + beat_start
INSTRUMENTS = 2 # number of instruments in midifile
T_PER_BEAT = 4


def t_to_time(midi_data, t):
    # t is the index of the data in length (T_PER_BEAT for t per beat)
    # for generating new midi file
    tick_per_beat = 220
    tick = int(t * tick_per_beat / T_PER_BEAT)
    time = midi_data.tick_to_time(tick)
    
    return time


def time_to_t(midi_data, time):
    # get t (T_PER_BEAT for t per beat) according to time
    beats = midi_data.get_beats()
    tick_per_beat = midi_data.time_to_tick(beats[len(beats) // 2]) // (len(beats) // 2)
    tick = midi_data.time_to_tick(time)
    t = round(tick * T_PER_BEAT / tick_per_beat)
    
    return int(t)


def get_file_paths(data_path):
    
    try:
        midi_files = [os.path.join(data_path, path) \
                      for path in os.listdir(data_path) \
                      if '.mid' in path or '.midi' in path]
    except OSError as e:
        print('Error: Invalid datapath!!!')
        
    print('{} midifiles found.'.format(len(midi_files)))
    
    return midi_files


def get_data_paths(data_path):
    
    try:
        data_files = [os.path.join(data_path, path) \
                      for path in os.listdir(data_path) \
                      if '.npy' in path]
    except OSError as e:
        print('Error: Invalid datapath!!!')
        
    print('{} data files found.'.format(len(data_files)))
    
    return data_files


def load_data(midi_file):
    # return data format: [inst, t, pitch]
    # return dimension: [INSTRUMENT, number_of_ts, NUMBER_FEATURES]
    
    print('----load data from midifile: ' + midi_file)
    
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    end_time = midi_data.get_end_time()
    number_of_ts = time_to_t(midi_data, end_time)
    
    data = np.zeros((INSTRUMENTS, number_of_ts, NUMBER_FEATURES), dtype=np.bool)
    data[:, :, NUMBER_FEATURES - 2] = 1 # rest
    for t in range(number_of_ts):
        if t % 4 == 0:
            data[:, t, NUMBER_FEATURES - 1] = 1 # beat_start
            
    for inst in range(INSTRUMENTS):
        onsets = []
        for note in midi_data.instruments[inst].notes:
            start_t = time_to_t(midi_data, note.start)
            end_t = time_to_t(midi_data, note.end)
            if start_t < end_t:
                pitch = note.pitch
                data[inst, start_t, pitch] = 1 # midi_note noset
                data[inst, start_t + 1 : end_t, NUMBER_FEATURES - 3] = 1 # sustain
                data[inst, start_t : end_t, NUMBER_FEATURES - 2] = 0 # rest
                onsets.append(start_t)
                
        for onset in onsets:
            data[inst, onset, NUMBER_FEATURES - 3] = 0 # not sustain
            
    return data


def to_monophonic(data):
    # return data format: [inst, t, pitch]
    # return dimension: [INSTRUMENT, number_of_ts, NUMBER_FEATURES]
    
    print('convert music into monophonic...')
    
    number_of_ts = len(data[0])
    
    # take the highest pitch for each instrument
    for inst in range(INSTRUMENTS):
        for t in range(number_of_ts):
            p = -1
            for pitch in range(127, -1, -1):
                if data[inst, t, pitch]:
                    p = pitch
                    break
            if p != -1:
                data[inst, t, :p] = 0


def to_octave(data_raw):
    # return data format: [inst, t, pitch]
    # return dimension: [INSTRUMENT, number_of_ts, NUMBER_FEATURES_OCTAVE]
    
    print('convert music into within one octave...')
    
    number_of_ts = len(data_raw[0])
    
    data = np.zeros((INSTRUMENTS, number_of_ts, NUMBER_FEATURES_OCTAVE), dtype=np.bool)
    
    for inst in range(INSTRUMENTS):
        for t in range(number_of_ts):
            for pitch in range(128):
                if data_raw[inst, t, pitch]:
                    data[inst, t, pitch % 12] = 1
            data[inst, t, 12:] = data_raw[inst, t, 128:]
            
    return data


def reverse_octave(data_octave):
    # return data format: [inst, t, pitch]
    # return dimension: [INSTRUMENT, number_of_ts, NUMBER_FEATURES]
    
    print('convert music within octave back into full piano roll...')
    
    number_of_ts = len(data_octave[0])
    
    data = np.zeros((INSTRUMENTS, number_of_ts, NUMBER_FEATURES), dtype=np.bool)
    
    for inst in range(INSTRUMENTS):
        for t in range(number_of_ts):
            for pitch in range(12):
                if data_octave[inst, t, pitch]:
                    if inst == 0:
                        data[inst, t, pitch + 12 * 5] = 1
                    else:
                        data[inst, t, pitch + 12 *4] = 1
            data[inst, t, 128:] = data_octave[inst, t, 12:]
    
    return data


def shift(data):
    # use it for data augmentation, shift the octave data by one semitone
    number_of_ts = len(data[0])
    
    line = np.zeros((INSTRUMENTS, number_of_ts, 1), dtype=np.bool)
    
    line[:, :, 0] = data[:, :, 0]
    data[:, :, :11] = data[:, :, 1:12]
    data[:, :, 11] = line[:, :, 0]
    

def load_data_all(data_path):
    # load all the data that converted into monophonic and within one octave
    # in the given midi_files...
    # return data shape:
    # (number_of_files)(INSTRUMENTS, number_of_ts_in_midi, NUMBER_FEATRUES_OCTAVE)
    
    midi_files = get_file_paths(data_path)
    data = []
    
    for midi_file in midi_files:
        data_cur = load_data(midi_file)
        to_monophonic(data_cur)
        data_cur = to_octave(data_cur)
        data.append(data_cur)
        for i in range(11):
            shift(data_cur)
            data.append(data_cur)
        
    print('done!')
    return data


def reload_data_all(data_path):
    # reload all the data that preloaded in monophonic and within one octave
    # in the given path in format .npy
    # return data shape:
    # (number_of_files)(INSTRUMENTS, number_of_ts_in_midi, NUMBER_FEATURES_OCTAVE)
    
    data_files = get_data_paths(data_path)
    data = []
    
    for data_file in data_files:
        data_cur = np.load(data_file)
        data.append(data_cur)
        for i in range(5):
            shift(data_cur)
        data.append(data_cur)
        for i in range(2):
            shift(data_cur)
        data.append(data_cur)
        
    print('done!')
    return data
        

def print_data(data):
    
    print('*' * 33)
    print('melody for the data is:')
    
    notes_per_line = 16
    number_of_ts = len(data[0])
    
    t = 0
    while t < number_of_ts - 1:
        for inst in range(INSTRUMENTS):
            for i in range(notes_per_line):
                onset = False
                for pitch in range(128):
                    if not onset and t < number_of_ts and data[inst, t, pitch]:
                        note_name = pretty_midi.note_number_to_name(pitch)
                        sys.stdout.write('{} '.format(note_name))
                        if len(note_name) == 2:
                            sys.stdout.write(' ')
                        onset = True
                if t < number_of_ts and data[inst, t, NUMBER_FEATURES - 3]:
                    sys.stdout.write('--  ')
                if t < number_of_ts and data[inst, t, NUMBER_FEATURES - 2]:
                    sys.stdout.write('00  ')
                t += 1
            if inst != INSTRUMENTS - 1:
                t -= notes_per_line
            print()
        print()


def generate_midi(data, filename):
    # generate midi file from data.
    print('*' * 33)
    print('generate midi...')
    
    number_of_ts = len(data[0])
    
    music = pretty_midi.PrettyMIDI()
    
    for i in range(INSTRUMENTS):
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)
        
        for t in range(number_of_ts):
            for pitch in range(128):
                if data[i, t, pitch]:
                    start_time = t_to_time(music, t)
                    end_t = t + 1
                    while end_t < number_of_ts and data[i, end_t, NUMBER_FEATURES - 3]:
                        end_t += 1
                    end_time = t_to_time(music, end_t)
                    # print('start_t:{}, end_t:{}'.format(t, end_t))
                    note = pretty_midi.Note(velocity=80, pitch=pitch, start=start_time, end=end_time)
                    piano.notes.append(note)
                    
        music.instruments.append(piano)
    music.write(filename)









