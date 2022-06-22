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

dbutils.fs.mount(
  source = "wasbs://container-example@databricstoragexsgxlyrd.blob.core.windows.net",
  mount_point = "/mnt/blob-storage",
  extra_configs = {"fs.azure.account.key.databricstoragexsgxlyrd.blob.core.windows.net":
                    dbutils.secrets.get(scope = "databricks-secret-scope",
                                        key = "blob-container-key")})

# COMMAND ----------

# df1 = spark.read.parquet("/mnt/blob-storage/mappings_gameid.parquet")
# df2 = spark.read.parquet("/mnt/blob-storage/mappings_team.parquet")

# COMMAND ----------

# df2 = df2.withColumn('row_num', f.monotonically_increasing_id()).withColumn("row_num", f.col("row_num").cast(StringType())).withColumn('name', f.concat(f.lit("Team "),f.col('row_num'))).withColumn('abbrev', f.concat(f.lit("Team "),f.col('row_num'))).drop('row_num')
# display(df2)

# COMMAND ----------

# df1 = spark.read.parquet("/mnt/blob-storage/mappings_team.parquet")

# COMMAND ----------

# df2 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/mrzutshi@microsoft.com/mappings_gameid-2.csv")

# COMMAND ----------

# df1.write.mode('overwrite').parquet("/mnt/blob-storage/mappings_gameid.parquet")
# df2.write.parquet("/mnt/blob-storage/mappings_team.parquet", mode="overwrite")

# COMMAND ----------

df = spark.read.parquet("/mnt/blob-storage/tracking_data.parquet")

# COMMAND ----------

# df = df.withColumn("dividerId", f.col("dividerId").cast(IntegerType()))
# df = df.withColumn("teamId", f.col("teamId").cast(IntegerType()))
# df = df.withColumn("period", f.col("period").cast(IntegerType()))
# df = df.withColumn("x_grid", f.col("x_grid").cast(IntegerType()))
# df = df.withColumn("y_grid", f.col("y_grid").cast(IntegerType()))
# df = df.withColumn("ball_x_grid", f.col("ball_x_grid").cast(IntegerType()))
# df = df.withColumn("ball_y_grid", f.col("ball_y_grid").cast(IntegerType()))
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
                             f.count("dx").alias("count"), #how many rows got grouped (dx is pretty meaningless here) any column thats not grouped by
                             f.sum(f.abs("dx")).alias("dx_sum"), 
                             f.sum(f.abs("dy")).alias("dy_sum"),
                             f.sum(f.abs("dx") + f.abs("dy")).alias("dxy_sum")) )
  
  
  return df_all_courts


# COMMAND ----------

df_all = {"courts": groupby_to_courts(df, velocity=False, divider=True),
         "v2": groupby_to_courts(df, velocity = False, divider = True, v2 = True),
          "balls": groupby_to_courts(df, velocity=False, divider=True, ball=True)}

# COMMAND ----------

additional_label = '0616'
if additional_label:
  additional_label = '_' + additional_label
else:
  additional_label = ''
  

#The below flags are really names for the experimental configuration, i.e. the "v2" experiment was decided to be [position, v2, ball], and the expbh experiement was [position, dx, dy, ball, expbh]
#Currently this is not configured to take in v2 AND expbh
v2 = False
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
