import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def shape_data(time_series:pd.Series, window_size:int=5)->Tuple[np.array,np.array]:
  """
  Takes a series and place the data in it into a matrix.
  The data is taken with a moving window of lenth 'window_size', then
  the 'window_size + 1' entry is taken as the value to predict.

  Inputs
  ------
  time_series: pd.Series
    A Pandas series containing a time series.
  window_size: int
    The rage of data needed to perform the prediction.

  """
  np_time_series = time_series.to_numpy()
  X = []
  y = []
  for i in range(len(np_time_series)-window_size):
    row = [[a] for a in np_time_series[i:i+5]]
    X.append(row)
    label = np_time_series[i+5]
    y.append(label)
  return np.array(X), np.array(y)


def data_split(
    time_series:pd.Series,
    window_size:int,
    train_size:float,
    val_size:float,
    random_state:int=0,
    shuffle:bool=False) -> Tuple[np.array,np.array,np.array,np.array,np.array]:
  """
  Takes a time series, apply the shape_data function on it and then split it
  into train,validation, and test sets.

  Inputs
  ------
  time_series: pd.Series
    A Pandas time series.
  window_size: int
    The rage of data needed to perform the prediction.
  train_size: float
    Percentage of the data that will be used to train the model.
  val_size floar:
    Percentage of the data that will be used in the validation process.
  random_state: int
    Number of the seed used to repeat the results of the model.
  shuffle: Boolean
    Arguments to stablish if shuffle the data or not.
  """
  X, y = shape_data(time_series)
  X_train, X_2, y_train, y_2 = train_test_split(X,y,train_size=train_size,random_state=random_state,shuffle=shuffle)
  X_val, X_test, y_val, y_test = train_test_split(X_2,y_2,train_size=val_size,random_state=random_state,shuffle=shuffle)
  return X_train,X_val,X_test,y_train,y_val,y_test
