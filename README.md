# automatic-music-accompaniment
Masters Final Project @QMUL, August 2018  
Student: Lele Liu  
Student Number: 130800488  
Supervisor: Emmanouil Benetos  
Student Email: lele.liu@se13.qmul.ac.uk or liulelecherie@gmail.com

## Project description
Here are the supporting materials for masters project **Automatic Music Accompaniment**. This poject uses deep learning LSTM neural network to predict and generate music accompaniment according to a given melody. The model is designed to include two LSTM layers and two time distributed dense layers. Dropout is used for the two dense layers. The model architecture can be seen <a href="https://github.com/cheriell/automatic-music-accompaniment/blob/master/images/model%20architecture.svg">here</a>.

## System reqirements
The project implementation uses python 3.6. Python package <a href="https://github.com/craffel/pretty-midi">PrettyMIDI</a> is needed for the data preperation for dealing with midi files. The deep model in this project is built using <a href="https://github.com/keras-team/keras">Keras</a> with <a href="https://www.tensorflow.org/">Tensorflow</a> backend. To run the code, please install Tensorflow first, and check to ensure you have the following python packages:

> <a href="https://github.com/keras-team/keras">keras</a> for building deep learning network   
> <a href="https://github.com/craffel/pretty-midi">pretty_midi</a> for processing midi data     
> <a href="https://github.com/numpy/numpy">numpy</a> for array functions     
> <a href="https://github.com/matplotlib/matplotlib">matplotlib</a> for ploting figures during model training

You would need ...... free memory for model training.

## File description
In the root folder, there are three trained model files _final_model.hdf5_, _final_model2.hdf5_, and _simple_model.hdf5_. The first two are models trained on a network with two LSTM and two dense layers, while the last one is trained on a simple network with one LSTM layer and one dense layer. Only classical music pieces are used in the training dataset of the three models, so the models would work best for predicting classical music.

The python codes are in _code_ folder, and some of the generated music pieces are provided in folder _generated music segments_.

## Running instructions
To use the code provided, please first save your midi dataset in a _midis_ folder in the following path format:     
_midis/composer/midifile.mid_     
Please make sure that all the midis you add in your dataset have two music parts.

Next, run _load_data_to_files.py_, this will encode the midis into data representations in _.npy_ format. The encoded musics will be monophonic and only contains two music parts.

After that, please create the following folders under _code_ folder:    
-_data_    
|---_train_    
|---_validation_   
Run file _divide_train_validation.py_, this will copy the encoded _.npy_ files into training and validation sets.

Add an _experiment_ folder under _code_, and use _train.py_ to train the complex model or _simple_model.py_ to train the simple model. The model training results will be saved in the created _experiment_ folder including models at the end of each epoch and figures for the losses and accuracies.

You can use _generate.py_ to generate music accompaniments. Run the file with command line options:   
_midifile.mid --model_file  model.hdf5 --diversity div_    
The diversity is in float format, and will be used in sampling notes in generating accompaniments. If you are using the provided model files _final_model.hdf5_ or _final_model2.hdf5_, the suggesting diversity is around 0.8. And if you are using _simple_model.hdf5_, you can try diversity around 0.6.




