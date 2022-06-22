# Databricks notebook source
# DBTITLE 1,Imports
import os
import string
import random
import numpy as np
import pickle

import matplotlib.pyplot as plt

import tensorflow as tf
print([tf.__version__, tf.config.list_physical_devices('GPU')])

from tensorflow import keras
from tensorflow.keras import backend as K
#import horovod.tensorflow.keras as hvd
#from sparkdl import HorovodRunner

import tensorflow.keras.layers as tkl
import tensorflow.keras.models as tkm
import tensorflow.keras.datasets as tkd

ALLCHARS = string.ascii_letters + string.digits

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Notebook Classes and Functions: 
# MAGIC - ###### NnUtils 
# MAGIC   - A class of basic shared utilities for Nueral Networks  
# MAGIC   - Attributes: 
# MAGIC               None
# MAGIC 
# MAGIC | Functions        | Arguments      |Description|
# MAGIC | -----------------| -------------- |-----------|
# MAGIC | **dense_reshape**    | x, dims, name      |Connects a layer to a dense fully connected linear layer and then reshapes to the desired output dims        |
# MAGIC | **postfix**  | name, postfix_length        | Adds a short random string to a string - Used to prevent name conflicts within a network            |
# MAGIC | **split_kwargs**| kwargs, args_list         |Splits a dictionary (kwargs) in to two dictionaries according to if the key is present in a list (args_list)           |
# MAGIC | **get_layers**| x, layers_config         |Creates a series of connected layers according to the dictionary layers_config |
# MAGIC | **create_basic_inputs**| cls, input_dim, name         |Creates an Input layer with shape: input_dim and name: name with a post-fix          |
# MAGIC 
# MAGIC 
# MAGIC - ###### Functions (freestanding)
# MAGIC 
# MAGIC | Functions        | Arguments      |Description|
# MAGIC | -----------------| -------------- |-----------|
# MAGIC | **get_data_mnist**    |flat       |returns dictionary of image and labels, each with nested train and test data      |
# MAGIC | **get_data_sin**  | timeseries_length, n_samples        | returns [n_samples, timeseries_length] so that each row is a sin wave           |
# MAGIC | **get_one_hot_map**| label_columns         |  Returns a dictionary one-hot mapping for the supplied list or vector           |
# MAGIC | **example_plot_single**| X_test, model_obj         |  Function to quickly plot an original heatmap against a prediction made with the supplied model |

# COMMAND ----------

def get_data_mnist(flat=True):
  '''
  A function to get, shape and return Mnist data for testing
  
  Parameters
  ----------
  Arguments:
   - None
  Keyword arguments:
   - flat -- (default=True) return images as vectors [1,n^2] (True) or matrices [n,n] (False)
   
  Returns
  ----------
  Dictionary of image and labels, each with nested train and test data
  '''
  
  (train_images, train_labels),(test_images, test_labels) = tkd.mnist.load_data()

  x_labels_train = np.zeros([train_labels.shape[0],10])
  for row, col in enumerate(train_labels):
    x_labels_train[row, col] = 1

  x_labels_test = np.zeros([test_labels.shape[0],10])
  for row, col in enumerate(test_labels):
    x_labels_test[row, col] = 1

  if flat:
    train_images = train_images.reshape(train_images.shape[0], train_images.shape[1]**2)/255.0
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1]**2)/255.0
    
  else:
    train_images = train_images/255.0
    test_images = test_images/255.0
    
  return {"images": {"train": train_images, "test": test_images},
          "labels": {"train_one_hot": x_labels_train, "test_one_hot": x_labels_test}}

def get_data_sin(timeseries_length=20, n_samples=5000):
    '''
    Get a matrix of sin waves
    Returns [n_samples, timeseries_length] so that each row is a sin wave
    '''
    
    x = np.linspace(0, 1000, timeseries_length*n_samples)

    return np.sin(x).reshape(timeseries_length,n_samples).transpose()

def get_one_hot_map(label_column):
  '''
  Returns a dictionary one-hot mapping for the supplied list or vector
  '''
  def one_hot(k,t):
    vec = np.zeros(t)
    vec[k] = 1
    return vec

  all_labels = set(label_column)
  count_labels = len(all_labels)

  return {label: one_hot(k, count_labels) for k, label in enumerate(all_labels)}

def example_plot_single(X_test, model_obj):
  '''
  Function to quickly plot an original heatmap against a prediction made with the supplied model
  '''
  n_test_samples = len(X_test)
  
  index = np.random.randint(n_test_samples)
  original_image = X_test[[index]]
  reconstructed_image = model_obj.predict(X_test[[index]])[0]

  original_image = original_image.reshape(28,28)
  reconstructed_image = reconstructed_image.reshape(28,28)

  plot_reconstruction(original_image, reconstructed_image)

# COMMAND ----------

# DBTITLE 1,Utility Class
class NnUtils:
  """
  A class of basic shared utilities for Neural Networks

  ...

  Attributes
  ----------
  None

  Methods
  -------
   - dense_reshape - Connects a layer to a dense fully connected linear layer and then reshapes to the desired output dims
   - _postfix - Adds a short random string to a string - Used to prevent name conflicts within a network
   - split_kwargs - Splits a dictionary (kwargs) in to two dictionaries according to if the key is present in a list (args_list) 
   - get_layers - Creates a series of connected layers according to the dictionary layers_config
   - create_basic_inputs - Creates an Input layer with shape: input_dim and name: name with a post-fix
  """
  
  def __init__(self):
    pass
  
  @classmethod
  def dense_reshape(cls, x, dims, name):
    '''
    Connects a layer to a dense fully connected linear layer and then reshapes to the desired output dims
    
    Arguments:
      x - the input layer
      dims - the desired output shape
      name - prefix for the two layers
    
    Returns:
      Output of Reshape layer (a tf layer)
    '''
    size = np.prod(dims)
    x = tkl.Dense(size, name=cls._postfix(name + "_dense"))(x)
    return tkl.Reshape(dims, name=cls._postfix(name + "_reshape"))(x)
  
  @staticmethod
  def _postfix(name, postfix_length=4):
    '''
    Adds a short random string to a string - Used to prevent name conflicts within a network
    
    Arguments:
      name - the string to which to apply the post-fix
      postfix_length (default=4) - length of the post-fix to be generated and joined
      
    Returns:
      A string with a post-fix attached
    '''
    if name[-1] != '_':
      name += "_"
    name += ''.join([ random.choice(ALLCHARS) for _ in range(postfix_length)])
    return name
  
  @staticmethod
  def split_kwargs(kwargs, args_list):
    '''
    Splits a dictionary (kwargs) in to two dictionaries according to if the key is present in a list (args_list)
    If the key is present in the args_list then the key-value pair is returned in in_dict
    If the key is not present in the args_list then the key-value pair is returned in not_in_dict
    
    Returns:
      in_dict, not_in_dict
    '''
    in_dict = {k: v for k, v in kwargs.items() if k in args_list}
    not_in_dict = {k: v for k, v in kwargs.items() if k not in args_list}
    return in_dict, not_in_dict
  
  @staticmethod
  def get_layers(x, layers_config):
    '''
    Creates a series of connected layers according to the dictionary layers_config
    The input to the first layer is x
    
    Returns:
      Output of last layer (a tf layer)
    '''
    for layer_config in layers_config:
      current_layer = getattr(tkl, layer_config["LayerType"])
      x = current_layer(**layer_config["LayerParameters"])(x)
    return x
  
  @classmethod
  def create_basic_inputs(cls, input_dim, name):
    '''
    Creates an Input layer with shape: input_dim and name: name with a post-fix
    
    Returns:
      Input layer according to spec (a tf layer)
    '''
    return tkl.Input(shape=input_dim, name=cls._postfix(name + "_image"))
    
#   def create_image_inputs(self, input_dim, name):
#     # INPUTS
#     inputs = tkl.Input(shape=(self.original_dim,), name=cls._postfix(name + "_image"))
#     return inputs, tkl.Reshape(self.image_dims, name=self._postfix(name + "_image_reshape"))(inputs)
  
#   def create_label_inputs(self, label_size, name):
#     # INPUTS
#     inputs = tkl.Input(shape=(self.label_size,), name=self._postfix(name + "_label_input"))
#     return inputs, self.dense_reshape(inputs, self.image_dims, name + "_label")

# COMMAND ----------

class ClassifierBase:
  '''Classifier Bases Class
  This class is to keep and maintain the common methods that can be 
  inherted by more spefic model types in the future.
  
  Examples are:
   - methods for common neural network blocks and patterns [bn_block]
   - methods for abstracting complicated fit loops [fit_]
  '''
  
  @staticmethod
  def get_bias(train_labels):
    '''Get bias_initializer and class weights for unbalanced datasets
    Inputs:
      train_labels - numpy or tensorflow array of training labels; expects a vector
      
    Outputs:
      bias_initializer (tf_constant), class_weight (dictionary)
    '''
    if tf.is_tensor(train_labels):
      train_labels = train_labels.numpy()
    
    neg, pos = np.bincount(train_labels.ravel())
    initial_bias = np.log([pos/neg])
    
    weight_for_0 = (1 / neg) * (len(train_labels) / 2.0)
    weight_for_1 = (1 / pos) * (len(train_labels) / 2.0)

    return tf.keras.initializers.Constant(initial_bias), {0: weight_for_0, 1: weight_for_1}
  
  @staticmethod
  def bn_block(layer_in, rate=0.1, activation='relu'):
    '''Add a batchnorm-dropout motif to your neural network
    The motif consists of:
     1. batch-normilzation
     2. Acticvation layer - default `relu`
     3. Dropout - default rate 0.1
    
    Inputs: 
     - layer_in is the tensor from the previous layer
   
    kwargs:
     - rate=0.1 is the dropout rate
     - activation='relu' is the activation for the activation layer
     
    Outputs:
     - layer_out is a tensor fro your to connect to subsequent layers
    '''
    x = tkl.BatchNormalization()(layer_in)
    x = tkl.Activation(activation=activation)(x)
    return tkl.Dropout(rate=rate)(x) #layer_out
  
  @staticmethod
  def fit_(dataset, class_weight, model_count, history, major_epochs=100, minor_epochs=1, data_frq=100):
    '''A common pattern for fitting the model
    
    Includes:
     1. Data refresh every `data_frq` (default 100 major epochs)
     2. Fit with train_images and train_labels; validate with test_images and test_labels (from the dict `dataset`)
     3. History is captured using the History class (arg `history`)
    
    Inputs:
      `major_epochs` is the number of the outer loop and the number of time the true model fit method is called
      `minor_epochs` is the classic "epochs" from the keras Model.fit() method
      `class_weight` (dict) is the results of .get_bias() method for unbalanced datasets
      
    Outputs:
      This function does not have a return
    '''

    for i in range(major_epochs):
      if i+1 % data_frq == 0:
        dataset = transfer_data.get_dataset()
      
      model_.fit(dataset['train_images'], tf.repeat(dataset['train_labels'],model_count+1,axis=1), 
                 validation_data=(dataset['test_images'], tf.repeat(dataset['test_labels'],model_count+1,axis=1)), 
                 epochs=minor_epochs, batch_size=transfer_data.desired_train_cases//100, verbose=False,
                 class_weight=class_weight)

      for index in range(model_count+1):
        history.capture_pr(model_, dataset['test_images'], dataset['test_labels'], index)

      history.capture_history(model_.history.history)

      print(f"Epoch: {i} -- APS={history.pr_curves[-(model_count+1)]['average_precision']}")
