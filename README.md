# automatic-music-accompaniment
Masters Final Project @QMUL, August 2018  
Student: Lele Liu  
Student Number: 130800488  
Supervisor: Emmanouil Benetos  
Student Email: lele.liu@se13.qmul.ac.uk or liulelecherie@gmail.com

## Description
Here are the supporting materials for masters project **Automatic Music Accompaniment**. This poject uses deep learning LSTM neural network to predict and generate music accompaniment according to a given melody. The model is designed to include two LSTM layers and two time distributed dense layers. Dropout is used for the two dense layers. The model architecture can be seen <a href="https://github.com/cheriell/automatic-music-accompaniment/blob/master/images/model%20architecture.svg">here</a>.

## System reqirements
The project implementation uses python 3.6. Python package <a href="https://github.com/craffel/pretty-midi">PrettyMIDI</a> is needed for the data preperation for dealing with midi files. The deep model in this project is built using <a href="https://github.com/keras-team/keras">Keras</a> with <a href="https://www.tensorflow.org/">Tensorflow</a> backend. To run the code, please install Tensorflow first, and check to ensure you have the following python packages:

> <a href="https://github.com/keras-team/keras">keras</a> for building deep learning network   
> <a href="https://github.com/craffel/pretty-midi">pretty_midi</a> for processing midi data     
> <a href="https://github.com/numpy/numpy">numpy</a> for array functions     
> <a href="https://github.com/matplotlib/matplotlib">matplotlib</a> for ploting figures during model training

You would need ...... free memory for model training.

## File description
In the root folder, there are three trained model files _final_model.hdf5_, _final_model2.hdf5_, and _simple_model.hdf5_. The first two are models trained on the network with two LSTM and two dense layers, while the last one is trained on the simple network with one LSTM layer and one dense layer.

The python codes are in _code_ folder, and some of the generated music pieces are provided in folder _generated music segments_.

## Running instructions
To use the code provided, please first save your midi dataset in midis/composer/midifile.mid

