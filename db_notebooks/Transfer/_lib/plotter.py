# Databricks notebook source
# MAGIC %run ../_data/model_data

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Notebook Classes: 
# MAGIC - ###### Plotter 
# MAGIC   - class that provides plotting and graphing functions for a dataset
# MAGIC   - Attributes: 
# MAGIC               inherits from sampleData,
# MAGIC               image_size = used to resize heatmaps to (n,n) for plotting
# MAGIC   
# MAGIC | Functions        | Arguments      |Description|
# MAGIC | -----------------| -------------- |-----------|
# MAGIC | **get_nearest_index**    | latent_mean, index         |Utility to find the row-index of the closest coordinates in latent space (nearest-neighbour)          |
# MAGIC | **quick_pair_plot**  | self, heatmap_name, index        |Plot a heatmap against its reconstructed counterpart (wrapper)      |
# MAGIC | **subplot**    | panel_coords, image, t_str        |Configures subplot          |
# MAGIC | **plot_reconstruction**| orginal_image, reconstructed_image, timeseries         |Plot a heatmap against its reconstructed counterpart (main code for plotting)          |
# MAGIC | **plot_play_images**  | self, index        | Plot a play as 4 component heatmaps     |
# MAGIC | **plot_latent_example**| self, index         |Plot a play as its coordinates in latent space relative to the cloud of other plays |
# MAGIC | **plot_candidates**| self, candidate_set, save_path         |Plot a central play as 4 component heatmaps and the heatmaps of similar plays          |

# COMMAND ----------

class Plotter(ModelData):
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
    
  def quick_pair_plot(self, heatmap_name='position', index=None):
    
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
    
  def plot_play_images(self, index=None):
    
    index = self.X_test.get_random_index(index)
      
    plt.figure(num=None, figsize=(16, 4), dpi=240, facecolor='w', edgecolor='k')
    for k, col_name in enumerate(self.col_names):
      self.subplot((1,4,k+1), self.X_test[col_name][index].reshape(self.image_size), col_name)

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
    plt.suptitle(title_str)
    
    print(title_str)
    
  def get_n_closest_plays(self, n, index=None, writeout = False):
    
    index = self.X_test.get_random_index(index)

    #print(self.z_test)
    print(index)
    #print(n)
    
    closest_index = self.get_nearest_n_index(self.z_test, index, n)
    
    
    print(closest_index)
    
    temp_df = X_test_labels.loc[np.append(index, closest_index), :].copy(deep = True)
    temp_df.loc[:, 'rank'] = np.arange(0, len(closest_index) + 1)
    
    if writeout:
      write_to = f'jf_cache_extra.{experiment_name}' + '_' + str(index)
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
