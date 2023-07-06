import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.engine.input_layer import InputLayer
from typing import Tuple

def create_lstm_model(input_shape:Tuple[int,int])-> Sequential:
  """
  Craete a nueral network with LSTM units.

  Inputs
  ------
  input_shape: Tuple[int,int]
    Shape of the input data.

  Output
  ------
    A Sequential model object.
  """
  model = Sequential()
  model.add(InputLayer(input_shape))
  model.add(LSTM(64))
  model.add(Dense(8,activation='tanh'))
  model.add(Dense(1,activation='linear'))

  print(model.summary())

  return model

