# Databricks notebook source
import cv2

import pyspark.sql.functions as f
from pyspark.sql.types import *

import numpy as npa

import pandas as pd
import random


from sklearn.mixture import BayesianGaussianMixture

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Notebook Classes (Fixed Rotation): 
# MAGIC - ###### SampleData 
# MAGIC   - class that stores data for Neural Network Models 
# MAGIC   - Attributes: 
# MAGIC               Data = data used to create instance (can be either np.ndarray or dict)
# MAGIC 
# MAGIC | Functions        | Arguments      |Description|
# MAGIC | -----------------| -------------- |-----------|
# MAGIC | **len**    | self         |returns number of samples in dataset          |
# MAGIC | **getitem**  | self, position        |returns data at specific posistion            |
# MAGIC | **keys**| self         |if data is of type dictionary, return distinct keys          |
# MAGIC | **values**| self         |if data is of type dictionary, returns list of all values in given dictionary |
# MAGIC | **items**| self         |if data is of type dictionary, returns key-value pairs as tuples in a list           |
# MAGIC | **get_random_index**| self, index, train_test (?)         |randomly generates an index from the dictionary |
# MAGIC | **get_data_as_list**| self         |returns dictionary data as a list to feed into models           |
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC - ###### GameLabels 
# MAGIC   - class that creates label/title that will be displayed with model results/heatmaps
# MAGIC   - Attributes: 
# MAGIC               label_columns = desired columns, 
# MAGIC               required_cols = required columns in table
# MAGIC 
# MAGIC | Functions        | Arguments      |Description|
# MAGIC | -----------------| -------------- |-----------|
# MAGIC | **validate**     | self, obj         |verifys that required columns are in the table, if not error with missing required column returned         |
# MAGIC | **complete_labels**              | self, maps        |returns teamNames, defenseNames, gcStart/End columns by mapping teamName and defenseName to the affiliated teams and also converting gcTime from total seconds to minutes format           |
# MAGIC | **get_row_labels**   | self, index         |returns row @ specified index with all of the columns in label_cols            |
# MAGIC | **get_row_title**   | self, title, long         | returns title (for heatmaps) that displays gameId, offensive/defensive team names, current period and gcTime  (NOT sure what long means in this context)   |
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC - ###### SampleDataSet 
# MAGIC   - Class that helps build model and sample latent space for similar plays
# MAGIC   - Attributes: 
# MAGIC                 X_train, X_training_labels, X_test, X_test_labels = Training/Testing data + labels,
# MAGIC                 X_hat_train, X_hat_test = predictions,  
# MAGIC                 z_train, z_test = latent space activations, 
# MAGIC                 z_train_prob, z_test_prob = posterior probabilities 
# MAGIC 
# MAGIC | Functions        | Arguments                               |Description|
# MAGIC | -----------------| ---------------------------- -----------|-----------|
# MAGIC | **emit_data**    |self                                     |returns X_training data + labels and X_testing data and labels     |
# MAGIC | **add_predictions** |self, model_obj                       |returns predictions for training and testing data (X_hat_train/test), the latent space activations (z_train/test), and posterior probability of latent space activation (z_train/test_prob)|
# MAGIC | **get_candidates** |self, latent_dim, n_neighbours, n_sets |returns candidate games/plays that are in close proximity to each other|
# MAGIC | **candidate_set_to_table** |self, candidate_set, save_path |returns corresponding rows/data based on selected candidates and saves output as table          |
# MAGIC 
# MAGIC 
# MAGIC - ###### ModelData
# MAGIC  - Class to generate dataset for models
# MAGIC  - Attributes: 
# MAGIC               experiment_name = experiment name,
# MAGIC               model_name = model name, 
# MAGIC               image_size = image size, 
# MAGIC               new_dividers = bool for whether new dividers are being used, 
# MAGIC               database_name = database name,
# MAGIC               h_rank = used for seed value, 
# MAGIC               flat = bool for whether data will be reshaped (image vs flat vector), 
# MAGIC               train_data_table = training set, 
# MAGIC               test_data_table = testing set,
# MAGIC               train_test_split = breakdown of what percent of data will be reserved for training/testing, 
# MAGIC               
# MAGIC 
# MAGIC | Functions        | Arguments      |Description|
# MAGIC | -----------------| -------------- |-----------|
# MAGIC | **generate_maps** |self           |creating maps for teamNames, teamIds, and gameIds         |
# MAGIC | **get_divider_maps**|self, df_heatmaps|returns map start and end times based on divider start/end times           |
# MAGIC | **write_data_to_table**|self          |merges previously generated tables (position, ball position, team latitude and team longitude), splits data into training and testing sets, samples and saves output             |
# MAGIC | **get_dataset** |self          |returns training and testing data + labels from the dataset           |
# MAGIC | **resize_courts** |self, df_heatmaps_pandas, col_name          |resizes heatmap images    |
# MAGIC | **cleanup_data** |self          |drops training and testing tables|

# COMMAND ----------

class SampleData:
  '''
  A class to store the data for the Neural Network models

  ...

  Attributes
  ----------
  - data - the data used to create the instance
    - May be either a np.ndarray or dict

  Methods
  -------
   - __len__ - Gets the number of samples in the data
   - __getitem__ - Used to access a sample of the data in a consistent way
   - split_kwargs - Splits a dictionary (kwargs) in to two dictionaries according to if the key is present in a list (args_list) 
   - get_random_index - return an index within the range of the data
   - get_data_as_list - return the data (dictionary) as a list ready to feed to a many-headed Neural Network
  '''
  def __init__(self, data):
    
    assert(type(data)) in [dict, np.ndarray]
    
    self.data = data
    
  def __len__(self):
    if type(self.data) is dict:
      xx = list(self.data.values())[0]
    else:
      xx = self.data
    
    #assert max(xx.shape) == xx.shape[0]
    return xx.shape[0]
    
  def __getitem__(self, position):
    return self.data[position]
  
  def keys(self):
    #doesn't work, probably not used in pipeline, should be self.data
    if type(data) is dict:
      return self.data.keys()
    else:
      return None
  
  def values(self):
    #doesn't work, probably not used in pipeline, should be self.data
    if type(data) is dict:
      return self.data.values()
    else:
      return None
  
  def items(self):
    #doesn't work, probably not used in pipeline, should be self.data
    if type(data) is dict:
      return self.data.items()
    else:
      return None
  #def get_model_data(self):
  #  return [x for x in self.data.values]
  
  def get_random_index(self, index, train_test='test'):
    if index is None:
      return np.random.randint(self.__len__())
    else:
      return index
  
  def get_data_as_list(self):
    return [x for x in self.data.values()]
    
  
@pd.api.extensions.register_dataframe_accessor("GameLabels")
class GameLabels:
  
  def __init__(self, pandas_obj):
    

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
  
  #TODO: Mrinal, the below had to be added so that the 'load' approach could also get row labels
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

class SampleDataSet:
  def __init__(self, X_train, X_train_labels, X_test, X_test_labels, maps, col_names, building_model_data = True):
    self.col_names = col_names
    
    self.X_train = SampleData(X_train)
    self.X_train_labels = X_train_labels.copy()#, maps)
    self.X_test = SampleData(X_test)
    self.X_test_labels = X_test_labels.copy()#, maps)
    
    self.X_train_labels.GameLabels.complete_labels(maps)
    self.X_test_labels.GameLabels.complete_labels(maps)
    
    self.X_hat_train = None
    self.X_hat_test = None
    
    self.z_train = None
    self.z_test = None
    
    self.z_train_prob = None
    self.z_test_prob = None
    
  def emit_data(self):
    return (self.X_train, self.X_train_labels), (self.X_test, self.X_test_labels)
  
  def add_predictions(self, model_obj):
    self.x_ = self.X_train.get_data_as_list()
    self.X_hat_train = SampleData(dict(zip(self.col_names, model_obj.predict(self.x_))))
    self.z_train = model_obj.encoder.predict(self.x_)
    
    bgm = BayesianGaussianMixture(n_components=1, random_state=42).fit(self.z_train)
    self.z_train_prob = np.exp(bgm.score_samples(self.z_train))
    
    self.x_ = self.X_test.get_data_as_list()
    self.X_hat_test = SampleData(dict(zip(self.col_names, model_obj.predict(self.x_))))
    self.z_test = model_obj.encoder.predict(self.x_)
    
    bgm = BayesianGaussianMixture(n_components=1, random_state=42).fit(self.z_test)
    self.z_test_prob = np.exp(bgm.score_samples(self.z_test))
    
  def get_candidates(self, latent_dim, n_neighbours=10, n_sets=20):
    base_nline = np.array([ -0.3, -0.15, 0.0, 0,15, 0.3 ])

    n = len(base_nline)

    total_coords = n ** latent_dim
    limit_coords = min(n_sets, total_coords)

    if total_coords > 10 ** 4:
      rand_i = np.random.randint(0, total_coords, limit_coords, dtype=np.long)
    else:
      rand_i = np.random.choice(range(total_coords), limit_coords, replace=False)

    all_coords = [ base_nline[i // (  n ** np.arange(latent_dim) ) % n] for i in rand_i]

    candidate_sets = []
    for v in all_coords:
      v = v.reshape(1, latent_dim)
      d = ((self.z_test - v) ** 2).sum(1)
      dx = np.argsort(d)
      cent = dx[0]
      neighbours = dx[1:n_neighbours+1]
      candidate_sets.append({"grid_coords": v,
                             "cent_dx": cent,
                             "cent_coods": self.z_test[cent,:],
                             "neigh_dx": neighbours,
                             "neigh_coords": self.z_test[neighbours,:]})
      
    return candidate_sets
    
  def candidate_set_to_table(self, candidate_set, save_path=None):
    n = len(candidate_set['neigh_dx'])
    c_df = self.X_test_labels.GameLabels.get_row_labels([candidate_set["cent_dx"]])
    n_df = self.X_test_labels.GameLabels.get_row_labels(candidate_set["neigh_dx"])
    df_candidates = pd.concat([c_df, n_df])

    df_candidates.loc[:, "is_center"] = [True] + [False] * n
    df_candidates.loc[:, "index"] = [candidate_set["cent_dx"]] + list(candidate_set["neigh_dx"])
    
    if save_path:
      #f"/dbfs/FileStore/gb-cache/experiments/{settings['experiment_name']}/{settings['model_name']}/candidates/candidates_{q}.csv"
      df_candidates.to_csv(save_path, index=False)
    
    return df_candidates
  

class ModelData:
  def __init__(self, experiment_name, image_size=(28,28), flat=True, new_dividers=False, database_name="default", h_rank=0, quick_run = False, forced_test_list = None, v2 = False, additional_label = None, expbh = False, chiral = False, chiral_list = None):
    
    #ModelData will take in the raw heatmaps generated by Generate Heatmaps notebook, rescale, reorganize, make a test and train split, assure forced list is in test set, and finally outputs that test and train into a specifc location
    
    #  Input from heatmaps:
    #  (position table) gameId, dividerId, [[court]]
    #  (v2 table) gameId, dividerId, [[court]]
    #  (ball table) gameId, dividerId, [[court]]
    #       -----> INIT modelData --->
    #  gameId, dividerId, [[position (normalized)]], [[v2 (normalized)]], [[ball (normalized)]]   
    # split into train and test sets and written to tables
    
    self.experiment_name = experiment_name
    #self.model_name = model_name
    self.image_size = image_size
    self.new_dividers = new_dividers
    self.database_name = database_name
    self.h_rank = h_rank
    self.flat = flat
    #Below is the manually labeled dividerIds that we want to be sure are in test set 
    self.forced_test_list = forced_test_list
    self.quick_run = quick_run
    #Type of experiment or heatmaps included
    self.v2 = v2
    self.expbh = expbh
    self.chiral = chiral
    self.chiral_list = chiral_list
    
    #JF Where the train and test tables will be written out to
    self.train_data_table = f"default.{self.experiment_name}_train" #_{self.model_name}
    self.test_data_table = f"default.{self.experiment_name}_test" #_{self.model_name}
    
    #JF spliting ratio hardcoded
    self.train_test_split = [0.7, 0.3]
    
    if additional_label:
      additional_label = '_' + additional_label
    else:
      additional_label = ''
      
      
    #The below should be rewritten to accomidate any combination of desired heatmaps, instead of specific combinations only
    #It is currently written according to the layout found in Generate Heatmaps, i.e. the "v2" experiment or the "expbh" experiment
    
    if self.v2:
      print(f'v2 turned on: {self.v2}')
      self.base_table_name = ("new_play_v2_positionmaps" + additional_label, "position")
      self.table_names = {"new_play_v2_velocitymaps" + additional_label: "v2", 
                          "new_play_v2_positionball" + additional_label: "ball"}
    #JF Only using new dividers now.  These are the locations where Heatmap Generation by DividerId places the heatmaps
    #Why is one of these a tuple and the other a dict?
    
    elif self.expbh:
      print(f'expbh turned on: {self.expbh}')
      self.base_table_name = ("new_play_positionmaps" + additional_label, "position")
      self.table_names = {"new_play_velocitymaps_dx" + additional_label: "dx", 
                          "new_play_velocitymaps_dy" + additional_label: "dy", 
                          "new_play_expbh" + additional_label: "expbh", 
                          "new_play_positionball" + additional_label: "ball"}
    
    elif self.new_dividers:
      self.base_table_name = ("new_play_positionmaps" + additional_label, "position")
      self.table_names = {"new_play_velocitymaps_dx" + additional_label: "dx", 
                          "new_play_velocitymaps_dy" + additional_label: "dy", 
                          "new_play_positionball" + additional_label: "ball"}
    else:
      self.base_table_name = ("play_positionmaps", "position")
      self.table_names = {"play_velocitymaps_dx": "dx", 
                          "play_velocitymaps_dy": "dy", 
                          "play_positionball": "ball"}
      
    self.col_names = [self.base_table_name[1]] + list(self.table_names.values())
    
    #The below does a lot more than write to table
    self.write_data_to_table(v2 = self.v2, expbh = self.expbh)
    
    self.generate_maps()
    
  
    
  def generate_maps(self):
    #PART OF INIT
    self.maps = {}
    
    self.maps['team_name_mapping_dict'] = \
      {
        x.nbaId: x.name for x in 
          spark.read.parquet("/mnt/blob-storage/mappings_team.parquet")
               .select("nbaId", "name")
               .collect()
       }
    
    self.maps['team_mapping_id_dict'] = \
      { 
        x.id: x.nbaId for x in 
          spark.read.parquet("/mnt/blob-storage/mappings_team.parquet")
               .select("nbaId", "id")
               .collect()
      }
    
    self.maps['game_mapping_dict'] = \
      { 
        x.gameId: [self.maps['team_mapping_id_dict'][x.homeTeamId], self.maps['team_mapping_id_dict'][x.awayTeamId]] for x in 
          spark.read.parquet("/mnt/blob-storage/mappings_gameid.parquet")
               .select(f.col("nbaId").alias("gameId"), "homeTeamId", "awayTeamId")
               .collect()
      }

  def get_divider_maps(self, df_heatmaps):
    
    def get_oh_list(k,n):
      x = [0] * n
      x[k] = 1
      return x

    all_starts = df_heatmaps.select(["StartDivider"]).distinct().collect()
    n = len(all_starts)
    all_starts = {x.StartDivider: get_oh_list(k,n) for k, x in enumerate(all_starts)}

    all_ends = df_heatmaps.select(["EndDivider"]).distinct().collect()
    n = len(all_ends)
    all_ends = {x.EndDivider: get_oh_list(k,n) for k, x in enumerate(all_ends)}

    map_start = f.udf(lambda start_divider: all_starts[start_divider], ArrayType(IntegerType()))
    map_end = f.udf(lambda end_divider: all_ends[end_divider], ArrayType(IntegerType()))

    return map_start, map_end
  
  def remove_shot_dividers(self, df_heatmaps):
    #PART OF INIT
    df_heatmaps = df_heatmaps.filter(f.col('StartDivider') != 'shot taken')
    return df_heatmaps


  def write_data_to_table(self, v2 = False, expbh = False):
    #PART OF INIT    
    #BULK OF COMPUTATION OF INIT MODEL_DATA takes place here
    
    #UDFs
    @udf("array<float>")
    
    def empty_heatmap():
      return [float(0.0) for x in np.arange(0,5151)]
    
    @udf("array<float>")
    def add_noise(xs):
      for i in range(1, len(xs)-1):
        #adds small amount of positive amplitude to neighboring cells in heatmap
        if ((xs[i] == 0) & ((xs[i-1] > 0)|(xs[i+1] > 0))):
          xs[i] = float(np.maximum(0.0, xs[i] + 1./10*(2*random.random() - 1)))  
        #could cals add noise to the nonzero part, *not sure* if necessary
      return xs
    
    def add_noise_to_dataframe(df, add_noise_to):
      for column in add_noise_to:
        df = df.withColumn(column, f.col(column))
        df = df.withColumn(column, add_noise(column))
      return df
    
    
    def transfer_games_to_test(forced_test_list, train_df, test_df, v2 = False, expbh = False, noise = True, identicals = True):
      
      #function that moves manually annotated games to test set and not the training set
      forced_test_df_pd = pd.DataFrame(forced_test_list)
      forced_test_df = spark.createDataFrame(forced_test_df_pd)
      
      rows_to_transfer = forced_test_df.alias('forced').join(train_df.alias('train'), (f.col('forced.gameId') == f.col('train.gameId')) 
                                                             & (f.col('forced.dividerId') == f.col('train.dividerId'))).select('train.*')
      rows_already_there = forced_test_df.alias('forced').join(test_df.alias('test'), (f.col('forced.gameId') == f.col('test.gameId')) 
                                                             & (f.col('forced.dividerId') == f.col('test.dividerId'))).select('test.*')

      print("{} subsequenced desired in test set were found in training set".format(rows_to_transfer.count()))
      print("{} subsequenced desired in test set were already in the test set".format(rows_already_there.count()))

      resulting_train_df = train_df.alias('train') \
        .join(rows_to_transfer.select('gameId', 'dividerId').alias('transfer'), (f.col('transfer.gameId') == f.col('train.gameId')) \
                                                                              & (f.col('transfer.dividerId') == f.col('train.dividerId') \
                                                                                ), "left_outer")\
                     .where(f.col('transfer.gameId').isNull()).select('train.*')
      
      print("{} # of columns in test_df".format(len(test_df.columns)))
      print("{} columns in test_df".format(test_df.columns))
      print("{} # of columns in rows_to_transfer".format(len(rows_to_transfer.columns)))
      print("{} columns in rows_to_transfer".format(rows_to_transfer.columns))
      print("{} # of columns in rows_already_there".format(len(rows_already_there.columns)))
      print("{} columns in rows_already_there".format(rows_already_there.columns))
      resulting_test_df = test_df.union(rows_to_transfer)
      
      both_df = rows_to_transfer.union(rows_already_there)
      
      if identicals:
        both_df = both_df.withColumn('dividerId', f.col('dividerId') + f.lit(10000)) 
        #simplest sanity check that divider id X and divider id 10000 + x are right next to each other  
        #[gameId:5, dividerId:120] is identical to [gameId:5, dividerId:10120] 
        resulting_test_df = resulting_test_df.union(both_df)
      
      if noise:

        if v2:
          noise_df = add_noise_to_dataframe(both_df, add_noise_to = ['position', 'v2', 'ball'])
        elif expbh:
          noise_df = add_noise_to_dataframe(both_df, add_noise_to = ['position', 'dx', 'dy', 'expbh', 'ball'])
        else:
          noise_df = add_noise_to_dataframe(both_df, add_noise_to = ['position', 'dx', 'dy', 'ball'])
          
        noise_df = noise_df.withColumn('dividerId', f.col('dividerId') + f.lit(20000)) #add 20000 to dividerId
        #e.g.  [gameId:5, dividerId:120] is the original, and [gameId:5, dividerId:30120] has the noise added to heatmaps according to add_noise_to_dataframe() 
        #TODO check what happens if identical, noise, and chiral are one make seure there isn't a collision between noise: 30XXX(i think) and chiral 20XXX
        resulting_test_df = resulting_test_df.union(noise_df)
        
      #returns the train_df minus the plays in forced_test_list
      #returns the test_df in additiona to plays in forced_test_list (that were found in train)
      #and the identical copies of those plays with dividerId 10XXX
      #and the noise added versions of those plays with dividerId 20XXX
      #(note that 30XXX is the noise_added version of the identical copies)
      return resulting_train_df, resulting_test_df, both_df
        
    
    #BODY OF FUNCTION
    
    select_columns = ["gameId", "teamId", "possId", "dividerId", "court"]
    on_columns = ["gameId", "teamId", "possId", "dividerId"]
    #below might not be needed with new dividerId mapping table
    #df_poss_basket_map = spark.sql("SELECT possId, basketX FROM nba_tracking_data.possessions").distinct()
    
    #Below follows the structure of the written out heatmap tables, e.g.
      #self.base_table_name = ("new_play_v2_positionmaps" + additional_label, "position")
      #self.table_names = {"new_play_v2_velocitymaps" + additional_label: "v2", 
      #                    "new_play_v2_positionball" + additional_label: "ball"}
    
    #BASE TABLE/HEATMAP (position)
    if self.quick_run:
      #quick_run is just to quickly make sure no errors
      forced_test = tuple(x['gameId'] for x in self.forced_test_list)
      df_heatmaps = ( spark.sql(f"SELECT * FROM {self.database_name}.{self.base_table_name[0]} where gameId in {forced_test}")
                         .withColumnRenamed("court", self.base_table_name[1]))
                         #.join(df_poss_basket_map, on='possId', how='left') )
    else:
      df_heatmaps = ( spark.sql(f"SELECT * FROM {self.database_name}.{self.base_table_name[0]}")
                         .withColumnRenamed("court", self.base_table_name[1]))
                         #.join(df_poss_basket_map, on='possId', how='left') )
      
    #JF ADD
    #Remove Play Subsequences that begin with a shot event (ball in air) for base table
    df_heatmaps = self.remove_shot_dividers(df_heatmaps)    
    
    #Displaying nulls for BASE TABLE
    temp_cols = ['teamId', 'startWcTime', 'endWcTime', 'startGcTime', 'endGcTime', 'period', 'gameId', 'possId', 'dividerId', 'outcome']
    df_heatmaps.select([f.count(f.when(f.isnan(c), c)).alias(c) for c in temp_cols]).show()
    df_heatmaps.select([f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in temp_cols]).show()
    df_heatmaps = df_heatmaps.where(f.col('possId').isNotNull())

    #Scale the heatmaps
    df_heatmaps = self.scale_court_column(df_heatmaps, self.base_table_name[1])
    print('Base Table done...')
    
                                    
    #ALL OTHER TABLES/HEATMAPS                                
    for table_name, new_column_name in self.table_names.items():
      print(f"Creating heatmap from {table_name}, for variable {new_column_name}")
      
      if self.quick_run:
        forced_test = tuple(x['gameId'] for x in self.forced_test_list)
        df_heatmaps = ( df_heatmaps.join(
                             spark.sql(f"SELECT * FROM {self.database_name}.{table_name} where gameId in {forced_test}")
                                  .select(select_columns)
                                  .withColumnRenamed("court", new_column_name), 
                             on=on_columns, how="left") )
      else:
        df_heatmaps = ( df_heatmaps.join(
                             spark.sql(f"SELECT * FROM {self.database_name}.{table_name}")
                                  .select(select_columns)
                                  .withColumnRenamed("court", new_column_name), 
                             on=on_columns, how="left") )
      
      current_null_ratio = float(df_heatmaps.filter(f.col(new_column_name).isNull()).count())/float(df_heatmaps.count() * 100) 
      print(f"% of missing velocity maps:{current_null_ratio}")
      
      #handling nulls (from left join, velocity maps missing? TODO find them)
      df_heatmaps = df_heatmaps.withColumn(new_column_name, 
                                           f.when(f.col(new_column_name).isNull(), 
                                                  empty_heatmap()).otherwise(f.col(new_column_name)))
      
      current_null_ratio = float(df_heatmaps.filter(f.col(new_column_name).isNull()).count())/float(df_heatmaps.count() * 100)             
      print(f"% of missing velocity maps after filling:{current_null_ratio}")
      
      #JF ADD
      #Removing shot dividers for all other tables (this shouldn't be needed as left join from base table which had these removed)
      df_heatmaps = self.remove_shot_dividers(df_heatmaps)
      
      temp_cols = ['teamId', 'startWcTime', 'endWcTime', 'startGcTime', 'endGcTime', 'period', 'gameId', 'possId', 'dividerId', 'outcome']
      # displaying nulls for ALL OTHER TABLES
      df_heatmaps.select([f.count(f.when(f.isnan(c), c)).alias(c) for c in temp_cols]).show()
      df_heatmaps.select([f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in temp_cols]).show()
      df_heatmaps = df_heatmaps.where(f.col('possId').isNotNull())

      #Scale the heatmaps
      df_heatmaps = self.scale_court_column(df_heatmaps, new_column_name)      
      print("Table: {} done...".format(table_name))

          
    map_start, map_end = self.get_divider_maps(df_heatmaps)

    #TODO: This shouldn't be needed as reason for new subsequence and end of subsequence come with df_preprocessed along with divider Id
    df_heatmaps = ( df_heatmaps.withColumn("StartDividerOH", map_start("StartDivider"))
                               .withColumn("EndDividerOH", map_end("EndDivider")) )

#     df_fta = spark.sql("SELECT possId, fta FROM nba_tracking_data.possessions")

#     df_heatmaps = df_heatmaps.join(df_fta, on="possId", how='left')

    df_heatmaps = ( df_heatmaps.filter(f.col("hasPoss") == True)
                               .filter(f.col("fta") == 0)
                               .filter(f.col("dividerId").isNull() == False) )
    
    #JF Here is where it is decided which data will be train and which will be test
    #After this would be a perfect spot to ensure that the subsequences you want to test are not in the train set, and are instead in the test set
    #But we need to make sure that every object upstream of this data reflects the change
    df_train, df_test = df_heatmaps.randomSplit(self.train_test_split, seed=42)
    print('# of train columns {}', format(len(df_train.columns)))
    print('# of test columns {}',format(len(df_test.columns)))
    print('train columns {}',format(df_train.columns))
    print('test columns {}',format(df_test.columns))
  
    #Switch em to test 
    if self.forced_test_list:
        df_train, df_test, both_df = transfer_games_to_test(self.forced_test_list, df_train, df_test, v2 = v2, expbh = expbh)
    
    if self.chiral:
        df_train, df_test, both_df = transfer_games_to_test(self.chiral_list, df_train, df_test, v2 = v2, expbh = expbh, noise = False, identicals = False)

    #TRAIN AND TEST SET WRITTEN OUT
    df_train.write.mode("overwrite").saveAsTable(self.train_data_table)
    df_test.write.mode("overwrite").saveAsTable(self.test_data_table)

#     if self.forced_test_list or self.chiral:
#       return df_train, df_test, both_df    
#     else:
#       return df_train, df_test  

  @staticmethod
  def scale_court_column(df_heatmaps, col_name):
        
    print('SCALING COURT for: {}'.format(col_name))
  
    def get_min_max(df_heatmaps, col_name):
      udf_max = f.udf(lambda x: float(np.max(x)), FloatType())
      udf_min = f.udf(lambda x: float(np.min(x)), FloatType())
    
      df_heatmaps = ( df_heatmaps.withColumn("positionMax", udf_max(col_name))
                                 .withColumn("positionMin", udf_min(col_name)) )

      min_max = df_heatmaps.agg(f.max("positionMax").alias("posMax"),
                                f.min("positionMin").alias("posMin")).collect()[0]
      x_min = min_max.posMin
      x_max = min_max.posMax

      df_heatmaps = df_heatmaps.drop("positionMax").drop("positionMin")#, f.col()])

      #assert x_min == 0

      return x_min, x_max

    x_min, x_max = get_min_max(df_heatmaps, col_name)
    print(f"{col_name} [Original Scale]: {x_min} -- {x_max}")

    udf_scaler = f.udf(lambda x_list: 
                         [ float(np.log( x + 1 ) / np.log( x_max + 1 )) for x in x_list ], 
                         ArrayType(FloatType()))

    df_heatmaps = df_heatmaps.withColumn(col_name, udf_scaler(col_name))

    x_min, x_max = get_min_max(df_heatmaps, col_name)
    print(f"{col_name} [Scaled]: {x_min} -- {x_max}")

    return df_heatmaps
    
  def get_dataset(self):
    
    #?JF: Why are we only sampling 40% of our datasets?

    
    df_train_pandas = ( spark.sql(f"SELECT * FROM {self.train_data_table}").toPandas())
                             #.sample(0.4, seed=self.h_rank).toPandas() )
    df_test_pandas = ( spark.sql(f"SELECT * FROM {self.test_data_table}").toPandas())
                             #.sample(0.4, seed=self.h_rank).toPandas() )

    X_train = {}
    X_test = {}
    
    print("Inside get_dataset, self.col_names {}".format(self.col_names))
    for col_name in self.col_names:
      print("Inside get_dataset, working on {}".format(col_name))
      X_train[col_name], X_train_labels = self.resize_courts(df_train_pandas, col_name)
      X_test[col_name], X_test_labels = self.resize_courts(df_test_pandas, col_name)

    #return X_train, X_train_labels, X_test, X_test_labels
      
    self.sample_data = SampleDataSet(X_train, X_train_labels, X_test, X_test_labels, self.maps, self.col_names)
      
    return self.sample_data.emit_data()


  
  def resize_courts(self, df_heatmaps_pandas, col_name, already_reoriented = False):
      
    #This thing basically cuts the court in half, and spins it around so all offensive courts are oriented in the same way. 
    #Again, this is coded in an obtuse way, and should be refactored to be human readable.

    #we'll need a new option in here for velocity phase space maps, to be warped to fit the same image, instead of just cut in half.  
    #this could be done up where v2 velocity gets scaled.  We could just fit it to image size right then.
    
    
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
      #if xi[:51,:].sum() > xi[50:,:].sum():
      
      if already_reoriented:
        if col_name == 'v2':
          Xi[c,:,:] = xi[25:76,:]
            
        else:
          Xi[c,:,:] = xi[:51,:]
        
        
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

    X_small = np.zeros([Xi.shape[0], *self.image_size, 1])

    for k in range(Xi.shape[0]):
      X_small[k,:,:,0] = cv2.resize(Xi[k,:,:], self.image_size)
      X_small[k,:,:,0] = X_small[k,:,:,0]/X_small[k,:,:,0].max()
      # Mrinal's addition
      X_small[np.isnan(X_small)] = 0
      
    if self.flat:
      X_small = X_small.reshape(len(df_heatmaps_pandas), np.prod(self.image_size))

    feature_columns = ["gameId", "dividerId", "teamId", "startWcTime", "endWcTime", "startGcTime", "endGcTime", "period"]
      
    return X_small, df_heatmaps_pandas[ feature_columns ]
  
  def cleanup_data(self):
    spark.sql(f"DROP TABLE {self.train_data_table}")
    spark.sql(f"DROP TABLE {self.test_data_table}")
    
  
    

# COMMAND ----------

from random import shuffle

sqlContext.sql("SET hive.mapred.supports.subdirectories=true")
sqlContext.sql("SET mapreduce.input.fileinputformat.input.dir.recursive=true")
sqlContext.sql("SET spark.sql.execution.arrow.pyspark.enabled=True")

class TransferData:
  def __init__(self, base_path="dbfs:/mnt/msft-nba-ml/rolling_courts/", 
               base_window_length=128, join_window_lengths=[], target_num_partitions=1, games_count=4,
               desired_train_cases=10000, desired_test_cases=5000):
    
    # Caputre the kwargs
    self.TARGET_NUM_PARTITIONS = target_num_partitions
    self.base_path = base_path
    self.base_window_length = base_window_length
    self.join_window_lengths = join_window_lengths
    self.desired_train_cases = 10000
    self.desired_test_cases = 5000
    
    # Defing the columns to be used to slice the data
    self.feature_columns = ['ballPositionHC', 'ballSpeedHC', 
                            'playerPositionHCDefense', 'playerSpeedHCDefense', 
                            'playerPositionHCOffense', 'playerSpeedHCOffense']
    self.label_columns = ['isPostUp']
    self.index_columns = ['gameId', 'period', 'possId', 'frameId']
    self.team_columns = ['teamIdOffense', 'teamIdDefense']
    self.cv_columns = ['is_test']
    
    # Get a list of data directories available to load; limited to the number of games `games_count`
    self.data_dirs = self.get_data_dirs(games_count)
    
    # Load the raw data from those directories
    self.raw_data, self.base_dataframe = self.load_data(self.data_dirs, self.TARGET_NUM_PARTITIONS * len(self.data_dirs), 
                                                        self.base_window_length, self.base_columns)
  
    # Line up the join_window_lengths with the base_window row-by-row for each frameId
    self.join_all_times()
    sepa
    self.get_counts_and_rates()
  
  @staticmethod
  def get_data_dirs_dict(required_windows={'128','64','32','16'}):

    data_dict = {}

    data_dirs = [x.path for x in dbutils.fs.ls('/mnt/msft-nba-ml/rolling_courts/')]

    shuffle(data_dirs)

    for data_dir in data_dirs:

      game_id = data_dir.split('-')[-1][:-1]
      window_length = data_dir.split('-')[-2]

      if game_id not in data_dict:
        data_dict[game_id] = {}

      data_dict[game_id][window_length] = data_dir

    return {k:v for k,v in data_dict.items() if set(v.keys()) == required_windows}
  
  @classmethod
  def get_data_dirs(cls, games_count):
    
    data_dirs_dict = cls.get_data_dirs_dict()
    
    some_data = {k:v for i, (k,v) in enumerate(data_dirs_dict.items()) if i < games_count}

    data_dirs = []
    for x in some_data.values():
      data_dirs += list(x.values())
      
    return data_dirs
  
  @classmethod
  def load_data(cls, data_dirs, target_num_partitions, base_window_length, base_columns):
    
    list_feature = None
    for data_dir in data_dirs:
      if list_feature:
        list_feature = list_feature.union(spark.read.parquet(data_dir).select(list_feature.columns))
      else:
        list_feature = spark.read.parquet(data_dir)

    raw_data = cls.add_train_test_label(list_feature)
        
    base_dataframe = raw_data.filter(f.col('window_length')==base_window_length).select(base_columns)
        
    return raw_data, base_dataframe

  @staticmethod
  def add_train_test_label(list_feature, test_split=1/3):
    
    distinct_game_poss = ( list_feature.select('gameId','possId').distinct()
                                       .withColumn('is_test', f.rand() > (1-test_split)) )

    return list_feature.join(distinct_game_poss, on=['gameId', 'possId'], how='left')
  
  @property
  def base_columns(self):
    return self.index_columns + \
           self.cv_columns + \
           self.team_columns + \
           [f.col(x).alias(f"{x}_w{self.base_window_length}_d0") for x in self.feature_columns] + \
           [f.col(x).alias(f"{x}_w{self.base_window_length}_d0") for x in self.label_columns]
  
  def get_join_columns(self, join_window_length, delay):
    
    return self.index_columns + \
           [f.col(x).alias(f"{x}_w{join_window_length}_d{delay}") for x in self.feature_columns] + \
           [f.col(x).alias(f"{x}_w{join_window_length}_d{delay}") for x in self.label_columns]
  
  def get_join_dataframe(self, join_window_length, delay):
    
    base_alias = f"df_{self.base_window_length}"
    join_alias = f"df_{join_window_length}"
    
    join_columns = self.get_join_columns(join_window_length, delay)
    
    return ( self.raw_data.filter(f.col('window_length')==join_window_length)
                                    .select(join_columns) )
    
  def get_sample_data(self):
    return \
      ( self.base_dataframe.filter(f.col('is_test')==False).sample(self.sample_rate_train).toPandas(),
        self.base_dataframe.filter(f.col('is_test')==True).sample(self.sample_rate_test).toPandas() )
    
  def get_dataset(self):
    def create_court(x):
      x = cv2.resize(np.array(x).reshape(51,51), (28,28))
      x /= x.max() + 10**-9
      return x.reshape(1,28,28,1)

    df_list_feature_train, df_list_feature_test = self.get_sample_data()
    
    train_images = [np.concatenate(df_list_feature_train[feature_column].apply(lambda x: create_court(x)).values, axis=0)
                    for feature_column in self.all_feature_columns]
    test_images = [np.concatenate(df_list_feature_test[feature_column].apply(lambda x: create_court(x)).values, axis=0)
                   for feature_column in self.all_feature_columns]

    train_labels = np.reshape(df_list_feature_train[self.all_label_columns].values, 
                              (len(df_list_feature_train), len(self.all_label_columns)))
    test_labels = np.reshape(df_list_feature_test[self.all_label_columns].values, 
                             (len(df_list_feature_test), len(self.all_label_columns)))

    train_index = df_list_feature_train[self.index_columns]
    test_index = df_list_feature_test[self.index_columns]

    ###### TEMP

    train_labels = train_labels[:,[0]].astype('bool')
    test_labels = test_labels[:,[0]].astype('bool')

    ###### Convert to Tensors for better memory management

    train_labels__tf = tf.convert_to_tensor(train_labels)
    test_labels__tf = tf.convert_to_tensor(test_labels)

    train_images__tf = [tf.convert_to_tensor(x) for x in train_images]
    test_images__tf = [tf.convert_to_tensor(x) for x in test_images]
    
    return {'train_images': train_images__tf,
            'test_images': test_images__tf,
            'train_labels': train_labels__tf,
            'test_labels': test_labels__tf,
            'train_index': train_index,
            'test_index': test_index}
    
  def join_all_times(self):
    
    base_alias = f"df_{self.base_window_length}"
    for join_window_length in self.join_window_lengths:
      
      for delay in range(0,self.base_window_length,join_window_length):
      
        join_alias = f"df_{join_window_length}_{delay}"

        join_dataframe = self.get_join_dataframe(join_window_length, delay)

        select_join = [f"{join_alias}.{x}" for x in join_dataframe.columns if x not in self.index_columns]

        self.base_dataframe = \
          ( self.base_dataframe.alias(base_alias)
             .join(join_dataframe.alias(join_alias),
                   on=( f.col(f'{base_alias}.gameId') == f.col(f'{join_alias}.gameId') ) & 
                      ( f.col(f'{base_alias}.possId') == f.col(f'{join_alias}.possId') ) &
                      ( f.col(f'{base_alias}.frameId') == f.col(f'{join_alias}.frameId') - delay ),
                   how='left')
             .select(f"{base_alias}.*",*select_join))
        
    self.clean_up_join()
        
  def clean_up_join(self):
    self.all_feature_columns = [x for x in self.base_dataframe.columns if "Position" in x or "Speed" in x]
    self.all_label_columns = [x for x in self.base_dataframe.columns if "isPostUp" in x]

    for feature_column in self.all_feature_columns:
      self.base_dataframe = self.base_dataframe.filter(f.col(feature_column).isNotNull())

    self.base_dataframe.cache()
    
  def get_counts_and_rates(self):
    self.total_samples = self.base_dataframe.count()
    
    self.is_test_counts = {x['is_test']: x['count'] for x in 
                           self.base_dataframe.select('is_test').groupBy('is_test').count().collect()}

    self.sample_rate_train = self.desired_train_cases/self.is_test_counts[False]
    self.sample_rate_test = self.desired_test_cases/self.is_test_counts[True]

# COMMAND ----------

if False:
  model_data = ModelData("test_mce", "test_mcm", new_dividers=True)
  (X_train, X_train_labels), (X_test, X_test_labels) = model_data.get_dataset()
  model_data.cleanup_data()

# COMMAND ----------


