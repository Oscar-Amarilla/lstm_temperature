import matplotlib.pyplot as plt
import numpy as np

def plot_history(history):
  """
  Plots the performance of the model in terms of the
  metric and the loss.

  Inputs
  ------
    history: keras.callbacks.History
      A keras callback containing the record of the 
      model's performance on each epoch.

  """
  plt.plot(history.history['mean_absolute_error'])
  plt.plot(history.history['val_mean_absolute_error'])
  plt.title('Model mean absolute error')
  plt.ylabel('mean absolute error')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

def hat_vs_test(y_hat:np.array,y_test:np.array):
  """
  Plots two graphs in one canvas: at the left, the curves of
  the test data and the predicted data of the model; at the right, 
  a correlation plot with a diagonal to visualize the quality 
  of the correlation.

  Inputs
  ------
    y_hat: np.array
      A numpy array containing the predicctions of the model
    y_test: np.array
      Test data.

  """
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5),layout='constrained', gridspec_kw={'width_ratios': [5, 2]})

  ax1.plot(y_test, label='Test data')
  ax1.plot(y_hat, label='Prediction')
  ax1.set_xlabel('Hours')
  ax1.set_ylabel('Temperature (ºC)')
  ax1.legend(loc='upper right')

  ax2.scatter(y_test, y_hat)
  ax2.plot(y_test,y_test,color='red')
  ax2.set_xlabel('Test data')
  ax2.set_ylabel('Prediction')

  fig.suptitle('Prediction vs Actual', fontsize=16);
