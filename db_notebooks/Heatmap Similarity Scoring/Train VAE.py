# Databricks notebook source
import cv2

import pyspark.sql.functions as f
from pyspark.sql.types import *

import numpy as npa
import copy
import pandas as pd
import random

# COMMAND ----------

# MAGIC %run ../Transfer/_lib/_models/base_autoencoder

# COMMAND ----------

# MAGIC %run ../Transfer/_data/model_data

# COMMAND ----------

# MAGIC %run ../Transfer/_lib/plotter

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Data
# MAGIC ### Model Data Initialization
# MAGIC The class `ModelData` is responsible for:
# MAGIC 1. Reading the database tables that resulted from the data processing pipeline
# MAGIC 2. Joining the different heatmaps so that a play has many heatmaps per row/sample
# MAGIC 3. Splitting the entire dataset into train and test sets
# MAGIC 4. Caching the train and test data to seperate temporary tables
# MAGIC 
# MAGIC ### Get Dataset Method
# MAGIC This is used to take samples from the cached train and test datasets
# MAGIC 1. To Prevent to much data being loaded in to VRAM per training period
# MAGIC 2. To allow samples to be drawn from separate workers when using Horovod (Distributed Training)
# MAGIC 
# MAGIC ### Result
# MAGIC 
# MAGIC Four data objects are produced:
# MAGIC 1. `X_train` - Training Data Dictionary - Each key is a image type (eg. position, dx, dy, ball) and the value is a n by (,28,28,1) matrix
# MAGIC 2. `X_train_labels` - Training Labels Pandas DataFrame - Labels associated with each play, per row of a data frame, n by (label-count [columns])
# MAGIC 1. `Y_train` - Testing Data Dictionary - Each key is a image type (eg. position, dx, dy, ball) and the value is a n by (,28,28,1) matrix
# MAGIC 1. `Y_train_labels` - Testing Labels Pandas DataFrame - Labels associated with each play, per row of a data frame, n by (label-count [columns])

# COMMAND ----------


# This Model data, will
# 1) read in the "heatmaps per player per frame" from tracking data.

run_name = '0620'

#MATCH TO HEATMAPS OUTPUT: below should match that chosen from Generate Heatmaps notebook so can find heatmaps
additional_label = '0620'

#Below doesn't matter as much, can make unique name for yourself
experiment_name = "full_results" + run_name

v2 = True
expbh = False
quick_run = False
chiral = False
#chiral_list = make_chiral_list(forced_test_list)
building_model_data = True

if building_model_data:
  model_data = ModelData(experiment_name, 
                        image_size=vgg_basic_architecture.stimulus_size[:2], new_dividers=True, flat=False, quick_run = quick_run, forced_test_list = None, v2 = v2, additional_label = additional_label, expbh = expbh, chiral = chiral)


# COMMAND ----------

#The following splits the code into to 1) if you've just built the model_data in the step above (i.e. building_model_data = True), versus 2) if you are pulling in previous train and test sets.  
# TODO FOR MRINAL: This is extremely redudant.  The functions should be refactored to take either way, instead of having copies of the functions from the class spelled out here. Ideally, the model_data class would just be loaded from before.  But if not, can set these functions up so that we don't have the same functions written twice.  Perhaps pull the functions out of the model_data class, and just have the model_data class USE those functions, with a flag for whether you're in load mode (and don't have acces to self (ModelData) or you have just built the model_data.


if building_model_data:
  (X_train, X_train_labels), (X_test, X_test_labels) = model_data.get_dataset()
else:
  
  
  #Everything in this 'else' is the hacky way to get what we need from previously written train and test tables
  #This is often the faster way to progress -- because building model_data is time consuming -- if the model_data has already been built before (and thus test and train tables written out)
  #This is a work around for not having the ability to save and load "model_data".

  maps = {}

  maps['team_name_mapping_dict'] = \
  {
    x.nbaId: x.name for x in 
      spark.sql("SELECT * FROM mappings.teams")
           .select(["nbaId", "name"])
           .collect()
   }

  maps['team_mapping_id_dict'] = \
  { 
    x.id: x.nbaId for x in 
      spark.sql("SELECT * FROM mappings.teams")
           .select(["nbaId", "id"])
           .collect()
  }

  maps['game_mapping_dict'] = \
  { 
    x.gameId: [maps['team_mapping_id_dict'][x.homeTeamId], maps['team_mapping_id_dict'][x.awayTeamId]] for x in 
      spark.sql("SELECT * FROM mappings.games")
           .select([f.col("nbaId").alias("gameId"), "homeTeamId", "awayTeamId"])
           .collect()
  }



  v2 = True
  if v2:
    print(f'v2 turned on: {v2}')
    base_table_name = ("new_play_v2_positionmaps" + additional_label, "position")
    table_names = {"new_play_v2_velocitymaps" + additional_label: "v2", 
                        "new_play_v2_positionball" + additional_label: "ball"}

  else: 
      base_table_name = ("new_play_positionmaps" + additional_label, "position")
      table_names = {"new_play_velocitymaps_dx" + additional_label: "dx", 
                          "new_play_velocitymaps_dy" + additional_label: "dy", 
                            "new_play_positionball" + additional_label: "ball"}

  col_names = [base_table_name[1]] + list(table_names.values())
  
  
  def resize_courts(df_heatmaps_pandas, col_name, already_reoriented = False):
      
    #This thing basically cuts the court in half, and spins it around so all offensive courts are oriented in the same way. 
    #Again, this is coded in an obtuse way, and should be refactored to be human readable.

    #we'll need a new option in here for velocity phase space maps, to be warped to fit the same image, instead of just cut in half.  
    #this could be done up where v2 velocity gets scaled.  We could just fit it to image size right then.
    image_size = (28,28)
    
    print(f"resizing for {col_name}")
    X = np.array(df_heatmaps_pandas[col_name].apply(lambda x: x.reshape(101,51)).tolist())
    print(X.shape)
    basketX = list(df_heatmaps_pandas["basketX"])
    
    
    # Starting with a column chosed from e.g. ['position', 'v2', 'ball]
    # [gameId-dividerId index, x_grid bin, y grid bin]
    # And then we'll need to only pick the half court we are interested in.
    # if we've previously reoriented by basketX in the heatmaps notebook, then this is simple.
    # (note v2 is different check out note above)
    Xi = np.zeros([X.shape[0], 51, 51])
    for c in range(X.shape[0]):
      xi = X[c,:,:]
      
      if already_reoriented:
        if col_name == 'v2':
          Xi[c,:,:] = xi[25:76,:]
            
        else:
          Xi[c,:,:] = xi[50:,:]
        
        
      else:
        if col_name == 'v2':
          #print("Different cut used for v2")
          if basketX[c] < 0:
            Xi[c,:,:] = xi[25:76,:]
          #TODO: BELOW IS WRONG Original error: pretty sure the below is a relfection for x coordinate, and not a rotation about z axis
          #still using it because of memory errors with the solution
          else:
            Xi[c,:,:] = xi[25:76,:][::-1]
          # THE BELOW IS CORRECT
  #         else:
  #           print('New Rotation')
  #           Xi[c,:,:] = np.array([x[::-1] for x in xi[25:76,:][::-1]])

        else:
          #print("Position cut used for {}".format(col_name))
          if basketX[c] < 0:
            Xi[c,:,:] = xi[:51,:]
          #TODO: BELOW IS WRONG Original error: pretty sure the below is a relfection for x coordinate, and not a rotation about z axis
          else:
            Xi[c,:,:] = xi[50:,:][::-1]
          #The Below is correct
          #else:
            #print('New Rotation')
            #Xi[c,:,:] = np.array([x[::-1] for x in xi[50:,:][::-1]])

    X_small = np.zeros([Xi.shape[0], *image_size, 1])

    for k in range(Xi.shape[0]):
      X_small[k,:,:,0] = cv2.resize(Xi[k,:,:], image_size)
      X_small[k,:,:,0] = X_small[k,:,:,0]/X_small[k,:,:,0].max()
      

    feature_columns = ["gameId", "dividerId", "teamId", "startWcTime", "endWcTime", "startGcTime", "endGcTime", "period"]
      
    return X_small, df_heatmaps_pandas[ feature_columns ]

  
  
  sample = False
  if sample:
    df_train_pandas = ( spark.sql(f"SELECT * FROM default.{experiment_name}_train")\
                               .sample(0.4, seed=0).toPandas() )
    df_test_pandas = ( spark.sql(f"SELECT * FROM default.{experiment_name}_test")\
                               .sample(0.4, seed=0).toPandas() )
  else:
    df_train_pandas = ( spark.sql(f"SELECT * FROM default.{experiment_name}_train").toPandas())
                            
    df_test_pandas = ( spark.sql(f"SELECT * FROM default.{experiment_name}_test").toPandas())
                         



  X_train = {}
  X_test = {}

  image_size = (28,28)
  flat = True

  print("Inside get_dataset, col_names {}".format(col_names))
  for col_name in col_names:
    print("Inside get_dataset, working on {}".format(col_name))
    X_train[col_name], X_train_labels = resize_courts(df_train_pandas, col_name, already_reoriented = True)
    X_test[col_name], X_test_labels = resize_courts(df_test_pandas, col_name, already_reoriented = True)

  sample_data = SampleDataSet(X_train, X_train_labels, X_test, X_test_labels, maps, col_names, building_model_data = building_model_data)


# COMMAND ----------

# df_train_pandas = ( spark.sql(f"SELECT * FROM default.{experiment_name}_train").toPandas())
# df_test_pandas = ( spark.sql(f"SELECT * FROM default.{experiment_name}_test").toPandas())

# COMMAND ----------

# DBTITLE 1,Create Model
latent_dim = 16

if building_model_data:
  vae_vgg_vae_var_single = AE(vgg_basic_architecture, experiment_name, model_data.col_names, latent_dim,
                             hvd_flag=False, variational=True)
else:
  vae_vgg_vae_var_single = AE(vgg_basic_architecture, experiment_name, col_names, latent_dim,    
                            hvd_flag=False, variational=True)
  
vae_vgg_vae_var_single.summary()

# COMMAND ----------

vae_vgg_vae_var_single.save_model()

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls -ltr /dbfs/FileStore/gb-cache/models/test_demo_experiment/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training
# MAGIC 
# MAGIC The purpose of is to find a trade of between reconstruction accuracy and a compressed representation of the input data - with greater compression coming from redcuding the number of nodes in the smallest layer - the latent space. We specify this with the parameter `latent_dim` and at the time of writing it was set to 4.
# MAGIC 
# MAGIC The reconstruction accuracy is the difference between the image, or images, that are used as input to the network and the images that are produced from the network. In our current network there are 4 images per sample: the position of the team `position`, the lateral speed of the team `dx`, the logditudinal speed of the team `dy`, and position of the ball `ball`.
# MAGIC 
# MAGIC These images are retreived from the database using the `ModelData` class. Within this object they are stored within the attribute `.sampledata` and leverage the `SampleDataSet` class. This stores the data in separate Trian and Test sets - under `.X_train` and `X_test`. These use the `SampleData` class and behave like a dictionary with extra custom features - so you can access the data using KEYS, eg.
# MAGIC 
# MAGIC ```python
# MAGIC model_data.sample_data.X_train.keys()                    # --> RETURNS dict_keys(['position', 'dx', 'dy', 'ball'])
# MAGIC ```
# MAGIC 
# MAGIC ```python
# MAGIC model_data.sample_data.X_train['position']               # --> RETURNS and np.ndarray
# MAGIC model_data.sample_data.X_train['position'].shape         # --> RETURNS (67868, 28, 28, 1)
# MAGIC ```
# MAGIC 
# MAGIC The last operation shows that there are 67868 training examples and that each example is an image of size (28, 28, 1). The 28 by 28 image size is dependent on the architecture of the model. With more layers and more pooling operations then the size of the image is increased to account for the loss pixels through the encoder. For example, in the `./models/architectures` notebook the `vgg_A_11_...` architecture requires images of the size 28 by 28 with 3 pooling layers; however the larger `vgg_A_19` with 4 pooling layers requires an image of the size 56 by 56.

# COMMAND ----------

# DBTITLE 1,Train The Model
test_epochs = 100

if building_model_data:
  x_train = model_data.sample_data.X_train.get_data_as_list()
  y_train = model_data.sample_data.X_train.get_data_as_list()
else:
  # #if import with lost model_data
  x_train = sample_data.X_train.get_data_as_list()
  y_train = sample_data.X_train.get_data_as_list()

# COMMAND ----------

if v2:
  dx = list(set(np.where(np.isnan(x_train[0].reshape(x_train[0].shape[0], 28*28)).sum(axis=1) == 0)[0].tolist()) & 
  set(np.where(np.isnan(x_train[1].reshape(x_train[1].shape[0], 28*28)).sum(axis=1) == 0)[0].tolist()) & 
  set(np.where(np.isnan(x_train[2].reshape(x_train[2].shape[0], 28*28)).sum(axis=1) == 0)[0].tolist()))
  
else:
  dx = list(set(np.where(np.isnan(x_train[0].reshape(x_train[0].shape[0], 28*28)).sum(axis=1) == 0)[0].tolist()) & 
  set(np.where(np.isnan(x_train[1].reshape(x_train[1].shape[0], 28*28)).sum(axis=1) == 0)[0].tolist()) & 
  set(np.where(np.isnan(x_train[2].reshape(x_train[2].shape[0], 28*28)).sum(axis=1) == 0)[0].tolist()) &
  set(np.where(np.isnan(x_train[3].reshape(x_train[3].shape[0], 28*28)).sum(axis=1) == 0)[0].tolist()))


x_train = [ x[dx] for x in x_train]
y_train = [ x[dx] for x in y_train]

# THIS IS THE FIT
vae_vgg_vae_var_single.fit(x_train, y_train, validation_split = 0.2, epochs=test_epochs)

# COMMAND ----------

plt.plot(vae_vgg_vae_var_single.model.history.history['loss'])
plt.plot(vae_vgg_vae_var_single.model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference and Results
# MAGIC 
# MAGIC The model is now trained on the X_train data. To test the usefulness of the model on finding related basketball plays we run inference and subsequent analyses only of the test data - X_test.
# MAGIC 
# MAGIC We run and record inference over the test data using the `.add_predictions` method of the `ModelData` class. This method captures:
# MAGIC 
# MAGIC 1. The predictions for (i) Train and (ii) Test data
# MAGIC   - (i) `.sample_data.X_hat_train` & (ii) `.sample_data.X_hat_test` - `SampleData` class
# MAGIC 2. The latent space activiations for (i) Train and (ii) Test data
# MAGIC   - (i) `.sample_data.z_train` & (ii) `.sample_data.z_test` - `np.ndarray` class - shape: `(n_samples, latent_dim)`
# MAGIC 3. The posterior probability of the latent space activations modelled as a latent_dim (dimensional) gaussian
# MAGIC   - (i) `.sample_data.z_train_prob` & (ii) `.sample_data.z_test_prob` - `np.ndarray` class - shape: `(n_samples, latent_dim)`
# MAGIC   
# MAGIC ### Model Flow Sketch
# MAGIC <img src ='/files/tables/ae_model_flow.PNG' width='70%'>

# COMMAND ----------

# DBTITLE 1,Run Inference and Add to DataSet
if building_model_data:
  model_data.sample_data.add_predictions(vae_vgg_vae_var_single)
else:
  sample_data.add_predictions(vae_vgg_vae_var_single)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Candidate Sets
# MAGIC 
# MAGIC A candidate set is a set of plays that are close to each other in latent space. The aim of the AutoEncoder method is that if two plays are compressed in to the latent space, so that their most important features are preserved during reconstruction by the decoder - then an efficient coding of those plays in latent space will lead to neighbours in that space baring greater similarity to each other than those that are found to be distant.
# MAGIC 
# MAGIC <img src ='/files/tables/latent_space.PNG' width='70%'>
# MAGIC 
# MAGIC A means to validate this is to find sets of latent space activations `model_data.sample_data.z_test` that are close to each other and consider them to be a "candidate set" - the plays that caused these activations are likely to have important common features and thereby be similar plays.
# MAGIC 
# MAGIC The converse should also be true: plays that are found in different candidate sets are found to be at greater distance to each other than those found within the set. In the diagram above the orange dots would be found n the same "candidate set", but the yellow dot would be found in another set (its own set with its own distinct neighbours). This means that the plays found in different sets most likely won't be too similar in terms of play similarity.
# MAGIC 
# MAGIC The method to generate these candidate sets belongs to the `ModelData` class - `.get_candidates`. It required the dimensionality of the latent space to define an n-dim cube within the latent space and to draw at most 20 samples from spaced points on the surface of the cube. This ensures that the distance between "candidate sets" is much larger than the distances between members of a particular set.

# COMMAND ----------

# DBTITLE 1,Find Plays That Are Close to Each Other
if building_model_data:
  candidate_sets = model_data.sample_data.get_candidates(vae_vgg_vae_var_single.latent_dim)
else:
  candidate_sets = sample_data.get_candidates(vae_vgg_vae_var_single.latent_dim)

# COMMAND ----------

candidate_sets

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Inspection
# MAGIC 
# MAGIC The `Plotter` class is a utility to quickly generate plots to inspect the data. Examples in this section:
# MAGIC 
# MAGIC 1. `plotter.plot_play_images()`
# MAGIC  - A method to plot a play as the 4 component heatmaps
# MAGIC 2. `plotter.quick_pair_plot()`
# MAGIC  - A method to plot a heatmap against its reconstructed counterpart
# MAGIC 3. `plotter.plot_latent_example()`
# MAGIC 4. `plotter.plot_closest_plays()`
# MAGIC 5. `plotter.plot_candidates()`

# COMMAND ----------

class Plotter(ModelData):
  
  #TODO: When satisfied with how this works, this needs to be moved to the model_data class
  '''
  A class to provide plotting and graphing functions for a dataset

  ...

  Attributes
  ----------
  - inherits from SampleData
  - Addtionally:
    - image_size - used to resize heatmaps to (n,n) for plotting
  
  Methods
  -------
   - get_nearest_index - Utility to find the row-index of the closest coordinates in latent space (nearest-neighbour)
   - quick_pair_plot - Plot a heatmap against its reconstructed counterpart (wrapper)
   - plot_reconstruction - Plot a heatmap against its reconstructed counterpart (main code for plotting)
   - plot_play_images - Plot a play as 4 component heatmaps
   - plot_latent_example - Plot a play as its coordinates in latent space relative to the cloud of other plays
   - plot_candidates - Plot a central play as 4 component heatmaps and the heatmaps of similar plays
  '''
  def __init__(self, model_data):
    self.__dict__ = model_data.sample_data.__dict__
    self.image_size = model_data.image_size

  @staticmethod
  def get_nearest_index(latent_mean, index):
    latent_distance = ((latent_mean - latent_mean[index,:]) ** 2 ).sum(1)
    return np.argsort(latent_distance)[1]
  
  @staticmethod
  def get_nearest_n_index(latent_mean, index, n):
    latent_distance = ((latent_mean - latent_mean[index,:]) ** 2 ).sum(1)
    return np.argsort(latent_distance)[1:n]
    
  def quick_pair_plot(self, heatmap_name='position', index=None): #NEED
    
    if not index:
      index = self.X_test.get_random_index(index)
    
    original_image = self.X_test[heatmap_name][index].reshape(28,28)
    reconstructed_image = self.X_hat_test[heatmap_name][index].reshape(28,28)
    
    self.plot_reconstruction(original_image, reconstructed_image)
    
  def subplot(self, panel_coords, image, t_str):
    plt.subplot(*panel_coords)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.title(t_str)
    
  def plot_reconstruction(self, original_image, reconstructed_image, timeseries=None):
    '''
    A function to get, shape and return Mnist data for testing

    Parameters
    ----------
    Arguments:
     - original_image - [n,n] matrix of the image used as input for the network
     - reconstructed_image - [n,n] matrix of the image produced as output from the network's predict method
    Keyword arguments:
     - timeseries -- (default=None) additional time series to plot as the third subplot

    Returns
    ----------
    No Return
    '''

    num_plots = 2 if timeseries is None else 3

    plt.figure(num=None, figsize=(8, 4), dpi=120, facecolor='w', edgecolor='k')

    self.subplot((1,num_plots,1), original_image, "Original Heatmap")
    self.subplot((1,num_plots,2), reconstructed_image, "Reconstructed Heatmap")

    if timeseries is not None:
      plt.subplot(1,num_plots,3)
      plt.plot(timeseries[0], linewidth=18)
      plt.plot(timeseries[1], linewidth=4, linestyle=':')
      plt.title("Timeseries")

    #_ = plt.suptitle("Colour Groups From Properties")
    plt.show()
    
  def plot_play_images(self, index=None): #NEED
    
    if not index:
      index = self.X_test.get_random_index(index)
      
    plt.figure(num=None, figsize=(16, 4), dpi=240, facecolor='w', edgecolor='k')
    for k, col_name in enumerate(self.col_names):
      self.subplot((1,5,k+1), self.X_test[col_name][index].reshape(self.image_size), col_name)

    row_title = self.X_test_labels.GameLabels.get_row_title(index)
    
    plt.suptitle(row_title)
    
  def plot_latent_example(self, index=None):
    
    index = self.X_test.get_random_index(index)
    
    latent_mean = self.z_test
    latent_prob = self.z_test_prob
    closest_index = self.get_nearest_index(latent_mean, index)

    plt.figure(num=None, figsize=(16, 4), dpi=240, facecolor='w', edgecolor='k')
    for k, dxs in enumerate([[0,1], [0,2], [1,2]]):
      i, j = dxs
      plt.subplot(1,3,k+1)
      
      plt.scatter(latent_mean[:, i], latent_mean[:, j], s=1, c=latent_prob)
      plt.scatter(latent_mean[index, i], latent_mean[index, j], s=60)
      plt.scatter(latent_mean[closest_index, i], latent_mean[closest_index, j], s=60)
      
      plt.xticks([])
      plt.yticks([])
      plt.title(f"Latent-D: {i}-{j}")
      
    plt.suptitle(f"Sample Probability: {latent_prob[index]:.2}")
    
  def plot_closest_plays(self, index=None):
    
    index = self.X_test.get_random_index(index)
    print(self.z_test)
    print(index)

    closest_index = self.get_nearest_index(self.z_test, index)
    
    print(closest_index)
    
    plt.figure(num=None, figsize=(16, 12), dpi=240, facecolor='w', edgecolor='k')
    for j, col_name in enumerate(self.col_names):
      self.subplot((3,4,j+1), self.X_test[col_name][index,:].reshape(28,28), f"Original {col_name}")
      self.subplot((3,4,j+1+4), self.X_test[col_name][closest_index,:].reshape(28,28), f"Closest Match {col_name}")
      self.subplot((3,4,j+1+2*4), self.X_hat_test[col_name][index,:].reshape(28,28), f"Reconstructed {col_name}")

    title_str = self.X_test_labels.GameLabels.get_row_title(index, title='Original') + '\n' + \
                self.X_test_labels.GameLabels.get_row_title(closest_index, title='Matches')
    plt.suptitle(title_str)
    
  def plot_closest_n_plays(self, n, index=None):
    
    index = self.X_test.get_random_index(index)

    #print(self.z_test)
    print(index)
    #print(n)
    
    closest_index = self.get_nearest_n_index(self.z_test, index, n)
    
    
    print(closest_index)
      
    plt.figure(num=None, figsize=(16, 12), dpi=240, facecolor='w', edgecolor='k')
    for i, ii in enumerate(closest_index):
      for j, col_name in enumerate(self.col_names):
        if i == 0:
          self.subplot((1 + len(closest_index),4,i+j+1), self.X_test[col_name][index,:].reshape(28,28), f"Original {col_name}")
        self.subplot((1 + len(closest_index), 4, i*4+j+1+4), self.X_test[col_name][ii,:].reshape(28,28), f"Rank {i + 1} closest {col_name}")
        #self.subplot((3,4,i+j+1+2*4), self.X_hat_test[col_name][index,:].reshape(28,28), f"Reconstructed {col_name}")

    title_str = self.X_test_labels.GameLabels.get_row_title(index, title='Original')
    for i, ii in enumerate(closest_index):
      title_str = title_str + '\n' + self.X_test_labels.GameLabels.get_row_title(ii, title=f'Rank {i}')
    #plt.suptitle(title_str)
    
    #print(title_str)
    
  def get_n_closest_plays(self, n, index=None, writeout = False, tag = None):  #NEED
    
    index = self.X_test.get_random_index(index)

    #print(self.z_test)
    print(index)
    #print(n)
    
    closest_index = self.get_nearest_n_index(self.z_test, index, n)
    
    
    print(closest_index)
    
    temp_df = X_test_labels.loc[np.append(index, closest_index), :].copy(deep = True)
    temp_df.loc[:, 'rank'] = np.arange(0, len(closest_index) + 1)
    
    if writeout:
      if tag:
        write_to = f'default.{experiment_name}' + '_' + str(index) + '_' + tag
      else:
        write_to = f'default.{experiment_name}' + '_' + str(index)
      print(f'Writing out table to: {write_to}')
      spark.createDataFrame(temp_df.loc[:, np.append('rank', X_test_labels.columns)]).write.mode("overwrite").saveAsTable(write_to)
    
    return temp_df.loc[:, np.append('rank', X_test_labels.columns)]
      
    

  def plot_candidates(self, candidate_set, save_path=None):
    '''
    Creates a figure to compare the heatmaps of similar plays
    '''
    def print_label(data_index, panel_index):
      '''
      Utility to plot the game information in the right-most panel of each row of the candidates figure
      '''
      t_str = self.X_test_labels.GameLabels.get_row_title(data_index, long=False)
      plt.subplot(11, 5, panel_index)
      plt.text(0, 0, t_str)
      plt.axis("off")
      
    fig = plt.figure(num=None, figsize=(14, 32), dpi=240, facecolor='w', edgecolor='k')
    for j, col_name in enumerate(self.col_names):
      self.subplot((11, 5, j+1), self.X_test[col_name][candidate_set["cent_dx"],:].reshape(self.image_size), f"Center {col_name}")
      print_label(candidate_set["cent_dx"], 5)

      for k, n_dx in enumerate(candidate_set['neigh_dx']):
        ki = k+1
        self.subplot((11, 5, j+1+5*ki), self.X_test[col_name][n_dx,:].reshape(self.image_size), f"Neighbour {ki} - {col_name}")
        if j == 3:
          print_label(n_dx, ki*5+5)

    plt.suptitle(f"Latent Coords Of Center: {candidate_set['grid_coords']}", y=0.9)

    if save_path:
      #f"/dbfs/FileStore/gb-cache/experiments/{settings['experiment_name']}/{settings['model_name']}/candidates/candidates_{q}.png"
      fig.savefig(save_path)
      
  def print_labels(self, n):
    print(self.X_test_labels.shape)
    for i in np.arange(0,n):
      print(self.X_test_labels.GameLabels.get_row_labels(i))
      
    return self.X_test_labels.GameLabels.get_row_labels(i)
  
  def find_index(self, gameId, divider_id):
    for i in np.arange(0, self.X_test_labels.shape[0]):
      if (self.X_test_labels.GameLabels.get_row_labels(i).gameId == gameId):
        if (self.X_test_labels.GameLabels.get_row_labels(i).dividerId == divider_id):
          print('index found in TEST: {}'.format(i))
          print(self.X_test_labels.GameLabels.get_row_labels(i))
          return i
    
    for i in np.arange(0, self.X_train_labels.shape[0]):
      if (self.X_train_labels.GameLabels.get_row_labels(i).gameId == gameId):
        if (self.X_train_labels.GameLabels.get_row_labels(i).dividerId == divider_id):
          print('index found in TRAIN: {}'.format(i))
          print(self.X_train_labels.GameLabels.get_row_labels(i))
          return None
    
        
    return None
  
  def temp_build_matrix(self, n):
    gameIds = []
    dividerIds = []
    periods = []
    gcstarts = []
    gcends = []
    for i in np.arange(0,n):
      gameId = self.X_test_labels.GameLabels.get_row_labels(i).gameId
      dividerId = self.X_test_labels.GameLabels.get_row_labels(i).dividerId
      gcstart = self.X_test_labels.GameLabels.get_row_labels(i).startGcTime    
      gcend = self.X_test_labels.GameLabels.get_row_labels(i).endGcTime
      period = self.X_test_labels.GameLabels.get_row_labels(i).period
      gameIds = gameIds + [gameId]
      dividerIds = dividerIds + [dividerId]
      periods = periods + [period]
      gcstarts = gcstarts + [gcstart]
      gcends = gcends + [gcend]
      
    return gameIds, dividerIds, periods, gcstarts, gcends
    
    

# COMMAND ----------

@pd.api.extensions.register_dataframe_accessor("GameLabels")
class GameLabels:
  
  def __init__(self, pandas_obj):
    

    #TODO: Mrinal, the below has been commented out because 'teamName' and 'denfenceName' are missing in the case of 
    self.label_cols = ['gameId', 'dividerId', 'teamId', 'teamName', 'defenceName', 'startGcTime', 'endGcTime', 'startWcTime', 'endWcTime', 'period']

    
    self.required_cols = ['gameId', 'teamId', 'startWcTime', 'endWcTime', 'startGcTime', 'endGcTime', 'period']
    self._validate(pandas_obj)
    self._obj = pandas_obj

  def _validate(self, obj):
    # verify there is a column latitude and a column longitude
    for col_name in self.required_cols:
      if col_name not in obj.columns:
        raise AttributeError(f"Must have '{col_name}'.")
      
  def complete_labels(self, maps):
    self._obj["teamName"] = self._obj["teamId"].apply(lambda team_id: maps['team_name_mapping_dict'][team_id])
    self._obj["defenceName"] = self._obj.apply(\
                                 lambda row: [ maps['team_name_mapping_dict'][x] 
                                                 for x in maps['game_mapping_dict'][row['gameId']] 
                                                 if x != row['teamId']][0], axis=1 )
    
    #self._obj["startWcTime"] = self._obj["startWcTime"].astype(int).apply(lambda x: f"{x//60:02d}:{x%60:02d}")
    #self._obj["endWcTime"] = self._obj["endWcTime"].astype(int).apply(lambda x: f"{x//60:02d}:{x%60:02d}")
    
    self._obj["startGcTime"] = self._obj["startGcTime"].astype(int).apply(lambda x: f"{x//60:02d}:{x%60:02d}")
    self._obj["endGcTime"] = self._obj["endGcTime"].astype(int).apply(lambda x: f"{x//60:02d}:{x%60:02d}")
      
  def get_row_labels(self, index):
    return self._obj.loc[index, self.label_cols]
  
  def get_row_labels_alt(self, index, cols):
    return self._obj.loc[index, cols]

  def get_row_title(self, index, title='Original', long=True):
    
    game_id, dividerId, team_id, team_name, defence_name, gc_start_time, gc_end_time, wc_start_time, wc_end_time, period = self.get_row_labels(index)
    
    if long:
      return \
      f'''{title} GameId: {game_id}; DividerId: {dividerId}, TeamId: {team_name} [hp] vs {defence_name}
      Time (P-{period}): {gc_start_time} - {gc_end_time}'''
    else:
      r_str = f"GameId: {game_id}\nDividerId: {dividerId}\nOffence: {team_name}\nDefence: {defence_name}\nPeriod {period}\n"
      r_str += f"Start GC: {gc_start_time}\nEnd GC: {gc_end_time}"
      return r_str

# COMMAND ----------

find_index(gameId = '0022000085', divider_id = 307, X_test_labels = X_test_labels, X_train_labels = X_train_labels)

# COMMAND ----------

# MAGIC %md
# MAGIC ### TODO FOR MRINAL:  if the plots below are showing with the basked at the bottom of the image, verify that the coordinates are correct from the court perspective, if so then just change how the plotter plots the heatmap so the basket is at the top.  TO REITERATE: double check FIRST that the x and y coordinates in the data, in the heatmaps, match video feed, and that the heatmaps aren't getting inverted.   Once that is VERIFIED, then you know it's just a plotter issue.

# COMMAND ----------

if building_model_data:
  plotter = Plotter(model_data)
  plotter.plot_play_images(index = 13)
else:
  plot_play_images(image_size, index = 13)

# COMMAND ----------

plotter.quick_pair_plot(heatmap_name='ball', index = 13)

# COMMAND ----------

plotter.plot_closest_n_plays(2, index=13)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting seed plays, or finding them, and then writing out similarity lists

# COMMAND ----------

plotter.get_n_closest_plays(20, index = 13 , writeout = True)

# COMMAND ----------


