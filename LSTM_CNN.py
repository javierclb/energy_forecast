import pandas as pd 
import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import (Concatenate, Dense, LSTM, Input, Dropout, LeakyReLU, 
                                     TimeDistributed, RepeatVector,
                                     ConvLSTM2D, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import numpy as np


def data_generator(data, backward, forward, mean, std):
    data = data.values
    N = len(data)
    normalized = ((data - mean)/std)
    target, _ = TimeseriesGenerator(normalized, normalized, length=forward, batch_size=N)[0]
    input, output = TimeseriesGenerator(normalized[:-forward],
                                 target, length=backward, 
                                 batch_size=N-forward)[0]
    return input, output

def get_train_test(data, backward, forward, samples):
    train = data[:-samples]
    test = data[-samples:]
    mean, std = train.mean(), train.std()
    input_train, output_train = data_generator(train, backward, forward, mean, std)
    input_test, output_test = data_generator(test, backward, forward, mean, std)
    
    return input_train, output_train, input_test, output_test, mean, std

def build_model(backward, forward, dropout=0):
    n, m = int(backward/24), int(backward/7)
    input_layer = Input(shape=(n, 1, m, 1))
    seq = ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu')(input_layer)
    seq = Flatten()(seq)
    seq = RepeatVector(24)(seq)
    seq = LSTM(units=200, dropout=dropout, return_sequences=True)(seq)
    seq = TimeDistributed(Dense(units=100, activation=tf.nn.relu))(seq)
    output_layer = TimeDistributed(Dropout(dropout))(seq)
    output_layer = TimeDistributed(Dense(1))(output_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
   
    return model


def train_model(input_train, output_train, epochs=10, dropout=0, train=True):
    n, bwd = input_train.shape
    fwd = output_train.shape[1]
    input_train = input_train.reshape(n, int(bwd/24), 1, int(bwd/7), 1)
    model = build_model(bwd, fwd, dropout=dropout)
    optimizer = Adam()
    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer)

    checkpoint_name = 'checkpoints/Weights.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, 
                            monitor='val_loss',
                            verbose = 1, 
                            save_best_only = True, 
                            mode ='min')
    
    if train:
        history = model.fit(input_train,
                    output_train,
                    epochs=epochs,
                    validation_split=0.3,
                    verbose=1,
                    callbacks=[checkpoint])
    
    else:
        pass
    
    model.load_weights("checkpoints/Weights.hdf5")

    return model

def model_predict(model, input_test, mean, std):
    input_test_cnn = input_test.reshape(input_test.shape[0], 7, 1, 24, 1)
    pred = model.predict(input_test_cnn) * std + mean
    return pred