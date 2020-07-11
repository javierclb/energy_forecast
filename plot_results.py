import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = len(history)
  num_out = len(true_future)
  error = mean_absolute_error(true_future, prediction)

  plt.plot(np.arange(num_in), history, label='History')
  plt.plot(np.arange(num_in, num_in + num_out),true_future, 'b--',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_in, num_in + num_out), prediction, 'r--',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.xlabel("Hours")
  plt.ylabel("Energy Demand MWh")
  plt.title(f"24 hour ahead forecast with error = {round(error, 2)}")
  #plt.ylim(0, 120)
  return plt