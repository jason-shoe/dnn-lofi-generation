import IPython
import pandas as pd
from scipy.io import wavfile

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import numpy as np
import math

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dropout,Dense, BatchNormalization, GRU
from tensorflow.keras import activations
from tensorflow.keras import regularizers


def to_sequences_new(dataset, input_len = 300, output_len = 150, stride = 25):
    '''
    input:
    dataset
        numpy 1D array
    intput_len
        integer
        how many samples for x
    
    
    '''
    x = []
    y = []

    for i in range(0,len(dataset)-input_len-output_len, stride):
        window = dataset[i:(i+input_len)]
        x.append(window)
        y.append(dataset[i+input_len:i+input_len+output_len])
        
    return np.array(x),np.array(y)

def to_sequences(dataset, input_len = 300, output_len = 150, stride = 25):
    '''
    input:
    dataset
        numpy 1D array
    intput_len
        integer
        how many samples for x
    
    
    '''
    x = []
    y = []

    for i in range(0,len(dataset)-input_len-output_len, stride):
        window = dataset[i:(i+input_len), 0]
        x.append(window)
        y.append(dataset[i+input_len:i+input_len+output_len, 0])
        
    return np.array(x),np.array(y)

def predict_song(model, scaler, x_in, num_iter, include_x = False,
                                        input_size = 300, output_size = 150):
    '''
    model
        tensorflow model
        function that is [batch_size, input_size, 1] -> [batch_size, output_size]
    x_in
        numpy array of size [input_size, 1]
        starting sample
    num_iter
        integer
        how many multiples of output_size that want to be generated
    include_x
        bool
        whether or not to include x in the beginning of the return
    input_size
        integer
        number of samples for input
    output_size
        integer
        numbe rof samples for output
    
    return:
    numpy array of shape [num_iter * output_size] or [num_iter * output_size + input_size]
    '''

    output = np.zeros(num_iter * output_size)
    curr_input = x_in   # must be of size [input_size, 1]
    for x in range(num_iter):
        curr_output = model.predict(np.array([curr_input]))[0]   # curr_output is size [150]
        output[x * output_size: (x + 1) * output_size] = curr_output
        curr_input = np.concatenate((curr_input, np.reshape(curr_output, (output_size, 1))), axis = 0)[-input_size:]
    
    if (include_x):
        to_return = np.concatenate((x_in.flatten(), output), axis = 0)
    else:
        to_return = output

    if (type(scaler) == int):
        return to_return * scaler
    return scaler.inverse_transform(to_return)