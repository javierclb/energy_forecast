import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np


def norm(x,train_stats):
    return (x - train_stats['mean']) / train_stats['std']


def prep_lstm(all_X,all_y,n,hours_prior):

	X_train, y_train = all_X[:-n], all_y[:-n]
	X_test, y_test = all_X[-n:],all_y[-n:]

	train_stats = pd.DataFrame(X_train).describe().transpose()

	X_train = norm(X_train,train_stats)
	X_test = norm(X_test,train_stats)
	input_lstm_train = X_train[["load"+str(i) for i in range(hours_prior)]]
	dense_input_train = X_train[X_train.columns[~X_train.columns.isin(input_lstm_train.columns)]]
	input_lstm_test = X_test[["load"+str(i) for i in range(hours_prior)]]
	dense_input_test = X_test[X_test.columns[~X_test.columns.isin(input_lstm_test.columns)]]

	#Estructura datos de entrenamiento
	shape = input_lstm_train.shape
	reshape_input = input_lstm_train.values.reshape(shape[0],shape[1],1)

	#Estructura datos de test
	shape_test = input_lstm_test.shape
	reshape_input_test = input_lstm_test.values.reshape(shape_test[0],shape_test[1],1)

	return reshape_input, dense_input_train, reshape_input_test, dense_input_test, shape, y_train


def LSTM_Model(all_X, all_y, EPOCHS=100,n=8760,hours_prior=24):
	
    reshape_input,dense_input_train,reshape_input_test,dense_input_test,shape,y_train=prep_lstm(all_X, all_y, n, hours_prior)

    ###############################Arquitectura red neuronal mixta##############################################

    first_input = Input(shape=(shape[1],1))
    seq = LSTM(units=20, dropout=0.2)(first_input)
    seq = Dense(40, activation=tf.nn.relu)(seq)
    seq = Dropout(0.3)(seq)
    second_input = Input(shape=(dense_input_train.shape[1], ))
    reg = Dense(40, activation=tf.nn.relu)(second_input)
    reg = Dropout(0.3)(reg)
    merged = Concatenate(axis=1)([seq, reg])
    output = Dense(80, activation=tf.nn.relu )(merged)
    #output=Dropout(0.3)(output)
    output = Dense(1)(output)

    model = Model(inputs=[first_input, second_input], outputs=output)
    optimizer = Adam()
    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=["mean_absolute_error", "mean_squared_error"],
    )

    model.summary()

    checkpoint_name = 'checkpoints/Weights.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='min')
    callbacks_list = [checkpoint]

    history = model.fit(
        [reshape_input, dense_input_train],
        y_train,
        epochs=EPOCHS,
        validation_split=0.3,
        verbose=1,
        callbacks=callbacks_list)

    #Cargar mejor modelo
    model.load_weights("checkpoints/Weights.hdf5")

    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=["mean_absolute_error", "mean_squared_error"],
    )

    return [float(f) for f in model.predict([reshape_input_test,dense_input_test])], model, history