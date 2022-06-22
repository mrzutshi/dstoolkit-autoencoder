# Databricks notebook source
from collections import Counter

import pickle
#import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
import scipy.cluster.hierarchy as sch
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from pyspark.sql import Row, SparkSession, Window


from pyspark.sql.types import StringType, IntegerType, LongType, FloatType, BooleanType, TimestampType, DataType


import pyspark.sql.functions as f
from pyspark.sql.functions import udf
from pyspark.sql.types import *

import random

import sys


# COMMAND ----------

# MAGIC %md
# MAGIC ## Sandbox

# COMMAND ----------

from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47")

# COMMAND ----------

from azure.identity import DefaultAzureCredential,AzureCliCredential, ChainedTokenCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

managed_identity = ManagedIdentityCredential()
azure_cli = AzureCliCredential()
credential_chain = ChainedTokenCredential(managed_identity, azure_cli)

secret_client = SecretClient(vault_url="https://rgautokeyvault.vault.azure.net/", credential=interactive_auth)

# COMMAND ----------

secret_client.get_secret("blob-container-key")

# COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://container-example@rgdbautodev001.blob.core.windows.net",
  mount_point = "/mnt/blob-storage",
  extra_configs = {"fs.azure.account.key.rgdbautodev001.blob.core.windows.net":
                    dbutils.secrets.get(scope = "databricks-secret-scope",
                                        key = "blob-container-key")})

# COMMAND ----------

df1 = spark.read.parquet("/mnt/blob-storage/mappings_gameid.parquet")
df2 = spark.read.parquet("/mnt/blob-storage/mappings_team.parquet")

# COMMAND ----------

#df = spark.read.format("csv").option("header", "true").load("/mnt/blob-storage/dividers.csv")

# COMMAND ----------

# df1.write.mode('overwrite').parquet("/mnt/blob-storage/mappings_gameid.parquet")
# df2.write.parquet("/mnt/blob-storage/mappings_team.parquet")

# COMMAND ----------

df = spark.read.parquet("/mnt/blob-storage/tracking_data.parquet")

# COMMAND ----------

# df = df.withColumn("divider_id", f.col("divider_id").cast(IntegerType()))
# df = df.withColumn("teamId", f.col("teamId").cast(IntegerType()))
# df = df.withColumn("period", f.col("period").cast(IntegerType()))
# df = df.withColumn("fta", f.col("fta").cast(IntegerType()))

# df = df.withColumn("ball_y", f.col("ball_y").cast(FloatType()))
# df = df.withColumn("x", f.col("x").cast(FloatType()))
# df = df.withColumn("ball_x", f.col("ball_x").cast(FloatType()))
# df = df.withColumn("y", f.col("y").cast(FloatType()))
# df = df.withColumn("gcTime", f.col("gcTime").cast(FloatType()))
# df = df.withColumn("dx", f.col("dx").cast(FloatType()))
# df = df.withColumn("dy", f.col("dy").cast(FloatType()))
# df = df.withColumn("dy", f.col("dividerId").cast(FloatType()))
# df = df.withColumn("ballDx", f.col("ballDx").cast(FloatType()))
# df = df.withColumn("ballDy", f.col("ballDy").cast(FloatType()))
# df = df.withColumn("basketX", f.col("basketX").cast(FloatType()))

# df = df.withColumn("wcTime", f.col("wcTime").cast(LongType()))

# df = df.withColumn("hasPoss", f.col("hasPoss").cast(BooleanType()))

# COMMAND ----------

display(df)

# COMMAND ----------

df.count()

# COMMAND ----------

# combined_df.write.mode('overwrite').parquet("/mnt/blob-storage/tracking_data.parquet")

# COMMAND ----------


def add_velocity_2(df, coord_list, time_coord, partition_by, scale_time = 1):
  """
  Add d_coord/d_time to the data set, grouped by partition_by
  
  e.g.
  df_tracking = add_velocity_2(df_tracking, ['x', 'y'], 'wcTime', ['gameId', 'playerId', 'period'], scale_time = 1/1000)
  
  TODO: should this be grouped by dividerId as well?
  """
  
  w = Window.partitionBy(partition_by).orderBy(time_coord)
  df = df.withColumn('d' + time_coord, (f.col(time_coord) - f.lag(time_coord, 1).over(w))*scale_time)
  for coord in coord_list:
    df = df.withColumn('d' + coord, f.round(f.col(coord) - f.lag(coord, 1).over(w),2)). \
      withColumn('v_' + coord, f.when(f.col(time_coord).isNull(), None).when(f.col('d' + coord) == 0, 0).otherwise(f.round(f.col('d' + coord)/f.col('d'+time_coord),4))).drop('d' + coord)
  #TODO: check why round to 4
  return df


def get_xy_data(df_tracking, velocity=False, divider=False, new_dividers=True, one_game=False, quick_run = False, v2 = False, restrict_to_games = None, expbh = False, reorient = True ):
  
  #This is where most of the computation takes place.  We get all of the tracking data
  
  col_names = "gameId, wcTime, gcTime, period, teamId, playerId, x, y, ball_x, ball_y, v_x, v_y, fta, basketX,reason_for_new_subsequence, reason_for_end_subsequence"
  
  
  if v2:
    df_tracking = add_velocity_2(df_tracking, ['x', 'y'], 'wcTime', ['gameId', 'playerId', 'period'], scale_time = 1/1000)
  

  if restrict_to_games:
    df_tracking = df_tracking.filter(f.col('gameId').isin(restrict_to_games))
    print(f'restricting to {len(restrict_to_games)}')
  
  if v2:
    tracking_cols = ['dividerId', 'ball_y', 'x', 'teamId', 'ball_x', 'wcTime', 'gameId', 'period', 'y', 'outcome', 'possId', 'hasPoss', 'gcTime', 'v_x', 'v_y', 'fta','basketX','reason_for_new_subsequence', 'reason_for_end_subsequence']
    
  else:
    tracking_cols = ['dividerId', 'ball_y', 'x', 'teamId', 'ball_x', 'wcTime', 'gameId', 'period', 'y', 'outcome', 'possId', 'hasPoss', 'gcTime', 'fta','basketX','reason_for_new_subsequence', 'reason_for_end_subsequence']
  
  #TODO: could just pull in this info from permanent written out divider_id mapping table
  df_tracking = ( df_tracking.filter(f.col("teamId") > 0)
                             #.withColumn("possId", udf_find_possId(f.col("gameId"), f.col("wcTime")))
                             .filter(f.col("possId").isNotNull()))
#                              .withColumn("hasPoss", udf_get_hasPoss(f.col("gameId"), f.col("possId"), f.col("teamId")))
#                              .withColumn("outcome", udf_get_outcome(f.col("gameId"), f.col("possId"), f.col("teamId"))) )
  
  #print(tracking_cols)
  
  df_tracking = df_tracking.withColumn('dividerId', f.col('divider_id'))
  df_tracking = df_tracking.drop('eventList')
  
  #COURT ROTATION
  #Reorient the positions so that the offense is always moving towards a positive x value.  This is a rotation about z-axis.   (x -> -x, y -> -y)
  #Everything derivied form position from here on out will be correct with respect to these new (rotated) coordinates.
  
  if reorient:  
    df_tracking = (df_tracking.withColumn('x', f.when(f.col('basketX') < 0, (-1)*f.col('x')).otherwise(f.col('x'))) 
                             .withColumn('y', f.when(f.col('basketX') < 0, (-1)*f.col('y')).otherwise(f.col('y')))
                             .withColumn('ball_x', f.when(f.col('basketX') < 0, (-1)*f.col('ball_x')).otherwise(f.col('ball_x')))
                             .withColumn('ball_y', f.when(f.col('basketX') < 0, (-1)*f.col('ball_y')).otherwise(f.col('ball_y'))))
  
  if velocity:
    print(df_tracking.schema.fields)
    df_tracking, vel_cols = add_velocity(df_tracking)
    tracking_cols += vel_cols

  if expbh:
    tracking_cols += ['playerId', 'has_ball']
    
  #If a new heatmap heatmap is built, restricts and a "grid" version of the feature would go here.
  just_xy = (df_tracking.select(tracking_cols)\
                         .withColumn("x_grid", f.round(f.col("x")).cast("Integer") )    
                         .withColumn("y_grid", f.round(f.col("y")).cast("Integer") )
                         .filter(f.abs(f.col("x_grid")) <= 50)
                         .filter(f.abs(f.col("y_grid")) <= 25)
                         .withColumn("ball_x_grid", f.round(f.col("ball_x")).cast("Integer") )
                         .withColumn("ball_y_grid", f.round(f.col("ball_y")).cast("Integer") )
                         .filter(f.abs(f.col("ball_x_grid")) <= 50)
                         .filter(f.abs(f.col("ball_y_grid")) <= 25))

  #NOTE! (MAX RESOLUTION OF IMAGES) here one is setting up for an aggregation at the integer level.  This sents a highest resolution possible to be 1 ft.  If you wanted more resolution in your heatmaps, you not only have to up it in the nn notebook, but also here upstream of VAE, as this sets the max resolution.
  
  if v2:
    #Note, the factor of (1/2.0) below is to make the v2 heatmap only cover the inner half of v_x grid. Our upper bound velocity will go to 51/2, blank grids will be padded out to 51, and later on we'll downsize by a factor of two, just like when we cut x in half, so that the image sizes will be the same as the position sizes.  See resize court function of model_data.
    #Using a "reasonable max", not absolute max, we don't care about getting extremely rare points in velocity space, but just the overall trend in velocity space, v_x_max = 25,  v_y_max = 15
    #With these maxes, we place into the factors below (inside the np.rounds)
    
    just_xy = just_xy.withColumn("v_x_grid", (f.lit(np.round((50.0/25.0*(1/2.0))))*f.col("v_x").cast("Integer"))).withColumn("v_y_grid", (f.lit(np.round(25.0/15.0))*f.col("v_y").cast("Integer")))

  if expbh:
    # we need to identify which players coordinates are apart of the expanded ball handler tracking.  This is defined as the players who either have the ball, or where they are 1.0 seconds before they have the ball, as well as 0.5 seconds after they lose the ball.  So there will be some player position overlap in time, and for the rest of it will be the ball handlers position.  1 second = 25 frames, 0.5 seconds ~ 12 frames.
    # so first we'll fill out a flag for the players that satisfy the above condition. 
    
    just_xy = fb_fill_n_rows(just_xy, ['gameId', 'playerId', 'dividerId'], 'wcTime', 'has_ball', 12, type = 'forward', new_column_name = 'expbh_flag_forward', convert_zeros_to_nulls = True)
    just_xy = fb_fill_n_rows(just_xy, ['gameId', 'playerId', 'dividerId'], 'wcTime', 'expbh_flag_forward', 25, type = 'backward', new_column_name = 'expbh_flag', convert_zeros_to_nulls = True)
    
    #T and second, we'll note the x and y grids for these flagged players only
    just_xy = just_xy.withColumn('x_grid_expbh', f.when(f.col('expbh_flag') == 1, f.col('x_grid')).otherwise(f.lit(None)))
    just_xy = just_xy.withColumn('y_grid_expbh', f.when(f.col('expbh_flag') == 1, f.col('y_grid')).otherwise(f.lit(None)))
    
    #we now have the x and y grid positions only of the expanded ball handlers, and everyone else null.
    
  #print(f"There are {just_xy.count()} rows in just_xy")
  
  #just_xy.cache()
  
  return just_xy, df_tracking, tracking_cols


# COMMAND ----------

#only configure these variables
quick_run = False
v2 = True
expbh = False

orig_just_xy, df_tracking, tracking_cols = get_xy_data(df, velocity=False, divider=True, new_dividers=True, one_game=False, quick_run = quick_run, v2 = v2, restrict_to_games = None, expbh = expbh)

# COMMAND ----------

display(orig_just_xy)

# COMMAND ----------


#Court Class
class Court:
  def __init__(self):
    
    self.schema_info = {"dividerId":
                          {"type": IntegerType(), "func": lambda x: x.iloc[0].dividerId},
                        "teamId":
                          {"type": IntegerType(), "func": lambda x: x.iloc[0].teamId},
                        "startWcTime": 
                          {"type": LongType(), "func": lambda x: min(x.start_wctime)},
                        "endWcTime": 
                          {"type": LongType(), "func": lambda x: max(x.start_wctime)},
                        "startGcTime": 
                          {"type": FloatType(), "func": lambda x: max(x.start_gctime)},
                        "endGcTime": 
                          {"type": FloatType(), "func": lambda x: min(x.start_gctime)},
                        "period": 
                          {"type": IntegerType(), "func": lambda x: x.iloc[0].period},
                        "gameId": 
                          {"type": StringType(), "func": lambda x: x.iloc[0].gameId},
                        "possId":
                          {"type": StringType(), "func": lambda x: x.iloc[0].possId},
#                         "wcDuration": 
#                           {"type": IntegerType(), "func": lambda x: x.iloc[0].wcDuration},
#                         "gcDuration": 
#                           {"type": IntegerType(), "func": lambda x: x.iloc[0].gcDuration},
                        "StartDivider": 
                          {"type": StringType(), "func": lambda x: x.iloc[0].StartDivider},
                        "EndDivider": 
                          {"type": StringType(), "func": lambda x: x.iloc[0].EndDivider},
                        "outcome": 
                          {"type": StringType(), "func": lambda x: x.iloc[0].outcome},
                        "hasPoss": 
                          {"type": BooleanType(), "func": lambda x: x.iloc[0].hasPoss},
                        "basketX":
                          {"type": FloatType(), "func": lambda x: x.iloc[0].basketX},
                        "fta":
                          {"type": IntegerType(), "func": lambda x: x.iloc[0].fta},
                        "court": 
                          {"type": ArrayType(FloatType()), "func": lambda x: x.values.ravel().tolist()}
                       }

  @property
  def output_func_dict(self):
    return {k: v["func"] for k,v in self.schema_info.items()}
  
  @property
  def output_schema(self):
    return StructType([StructField(k, v["type"]) for k,v in self.schema_info.items()])
    
  @staticmethod
  def get_court(data, output_func_dict, agg_col_name="count", divider=False, ball=False, v2 = False, expbh = False):

    if v2:
      df_court = data[["v_x_grid", "v_y_grid", agg_col_name]].pivot(
                  index='v_x_grid', columns='v_y_grid', values=agg_col_name).fillna(0.0)
    elif ball:
      df_court = data[["ball_x_grid", "ball_y_grid", agg_col_name]].pivot(
                  index='ball_x_grid', columns='ball_y_grid', values=agg_col_name).fillna(0.0)
    elif expbh:
      df_court = data[["x_grid_expbh", "y_grid_expbh", agg_col_name]].pivot(
                  index='x_grid_expbh', columns='y_grid_expbh', values=agg_col_name).fillna(0.0)  
    else:
      df_court = data[["x_grid", "y_grid", agg_col_name]].pivot(
                  index='x_grid', columns='y_grid', values=agg_col_name).fillna(0.0)

    # We can use the same set up here, as we have already scaled our velocities to fit withint the range that position fits in.
    missing_indexes = [x for x in list(range(-50,51)) if x not in df_court.index.tolist()]
    missing_columns = [x for x in list(range(-25,26)) if x not in df_court.columns.tolist()]

    for missing_index in missing_indexes:
      df_court.loc[missing_index,:] = 0.0

    for missing_column in missing_columns:
      df_court.loc[:,missing_column] = 0.0

    df_court = df_court.reindex(index=list(range(-50,51)))
    df_court = df_court.reindex(columns=list(range(-25,26)))
    
    print(df_court.count())
    #return df_court
    return pd.DataFrame([{**{k: v(data) for k,v in output_func_dict.items() if k != "court"},
                         **{k: v(df_court) for k,v in output_func_dict.items() if k == "court"}}])

# COMMAND ----------

def groupby_to_courts(just_xy, velocity=False, divider=False, ball=False, v2 = False, expbh = False):
  
  if v2:
    groupby_cols = ["gameId", "possId", "teamId", "hasPoss", "outcome", "dividerId", "v_x_grid", "v_y_grid", "fta", "basketX"]
  elif ball:
    groupby_cols = ["gameId", "possId", "teamId", "hasPoss", "outcome", "dividerId", "ball_x_grid", "ball_y_grid","fta", "basketX"]
  elif expbh:
    groupby_cols = ["gameId", "possId", "teamId", "hasPoss", "outcome", "dividerId", "x_grid_expbh", "y_grid_expbh" ,"fta", "basketX"]
  else:
    groupby_cols = ["gameId", "possId", "teamId", "hasPoss", "outcome", "dividerId", "x_grid", "y_grid", "fta", "basketX"]
  
  
  df_all_courts = ( just_xy.groupBy(groupby_cols)
                       .agg( f.min("wcTime").alias("start_wctime"),
                             f.max("wcTime").alias("end_wctime"),
                             f.max("gcTime").alias("start_gctime"),
                             f.min("gcTime").alias("end_gctime"),
                             f.first("period").alias("period"),
                             f.first("reason_for_new_subsequence").alias("StartDivider"),
                             f.first("reason_for_end_subsequence").alias("EndDivider"),
                             f.count(f.col('*')).alias('count')))

  
  
  return df_all_courts


# COMMAND ----------

df_all = {"courts": groupby_to_courts(orig_just_xy, velocity=False, divider=True),
         "v2": groupby_to_courts(orig_just_xy, velocity = False, divider = True, v2 = True),
          "balls": groupby_to_courts(orig_just_xy, velocity=False, divider=True, ball=True)}

# COMMAND ----------

additional_label = '0620'
if additional_label:
  additional_label = '_' + additional_label
else:
  additional_label = ''
  

#The below flags are really names for the experimental configuration, i.e. the "v2" experiment was decided to be [position, v2, ball], and the expbh experiement was [position, dx, dy, ball, expbh]
#Currently this is not configured to take in v2 AND expbh
v2 = True
expbh = False


court_obj = Court()


if v2:
  save_jig = \
           [
              {"col_name": "count", "table_name": "new_play_v2_positionmaps" + additional_label, "input": "courts", "divider": True, "ball": False, 'v2': False, 'expbh': False},
              {"col_name": "count", "table_name": "new_play_v2_velocitymaps" + additional_label, "input": "v2", "divider": True, "ball": False, 'v2': True, 'expbh': False},
              {"col_name": "count", "table_name": "new_play_v2_positionball" + additional_label, "input": "balls", "divider": True, "ball": True, 'v2': False, 'expbh': False}
           ]

elif expbh:
  save_jig = \
           [
              {"col_name": "count", "table_name": "new_play_positionmaps" + additional_label, "input": "courts", "divider": True, "ball": False, 'v2': False, 'expbh': False},
              {"col_name": "dx_sum", "table_name": "new_play_velocitymaps_dx" + additional_label, "input": "courts", "divider": True, "ball": False, 'v2': False, 'expbh': False},
              {"col_name": "dy_sum", "table_name": "new_play_velocitymaps_dy" + additional_label, "input": "courts", "divider": True, "ball": False, 'v2': False, 'expbh': False},
              {"col_name": "count", "table_name": "new_play_expbh" + additional_label, "input": "expbh", "divider": True, "ball": False, 'v2': False, 'expbh': True},
              {"col_name": "count", "table_name": "new_play_positionball" + additional_label, "input": "balls", "divider": True, "ball": True, 'v2': False, 'expbh': False}
           ]
  
else:
  save_jig = [
              {"col_name": "count", "table_name": "new_play_positionmaps" + additional_label, "input": "courts", "divider": True, "ball": False, 'v2': False, 'expbh': False},
              {"col_name": "dx_sum", "table_name": "new_play_velocitymaps_dx" + additional_label, "input": "courts", "divider": True, "ball": False, 'v2': False, 'expbh': False},
              {"col_name": "dy_sum", "table_name": "new_play_velocitymaps_dy" + additional_label, "input": "courts", "divider": True, "ball": False, 'v2': False, 'expbh': False},
              {"col_name": "dxy_sum", "table_name": "new_play_velocitymaps_dxy" + additional_label, "input": "courts", "divider": True, "ball": False, 'v2': False, 'expbh': False},
              {"col_name": "count", "table_name": "new_play_positionball" + additional_label, "input": "balls", "divider": True, "ball": True, 'v2': False, 'expbh': False}
           ]



for jig in save_jig:
  partial_get_court = partial(court_obj.get_court, output_func_dict=court_obj.output_func_dict,
                                                   agg_col_name=jig["col_name"], 
                                                   divider=jig["divider"], 
                                                   ball=jig["ball"],
                                                   v2=jig["v2"],
                                                   expbh=jig["expbh"])

  df_heatmaps = (df_all[jig["input"]].groupBy("teamId", "possId", "dividerId")
                                      .applyInPandas(partial_get_court, schema=court_obj.output_schema) )

  
  df_heatmaps.write.mode("overwrite").saveAsTable(f'default.{jig["table_name"]}')  #this writes out heatmaps, comment out if you want to play with these

# COMMAND ----------

display(df_heatmaps)

# COMMAND ----------

df_heatmaps.count()

# COMMAND ----------


