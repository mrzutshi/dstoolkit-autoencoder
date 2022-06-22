# Databricks notebook source
# MAGIC %md
# MAGIC #How to GENERATE HEATMAPS, train a VAE, and RETURN LIST OF SIMILAR PLAYS

# COMMAND ----------

# MAGIC %md
# MAGIC The purpose of this guide is for me to throw all of my thoughts somewhere, whilst guiding NBA/MSFT users on how to use this set up.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prerequisites
# MAGIC 
# MAGIC It is assumed that the following are available/have been completed:
# MAGIC   - a "df_preprocessed" has been created via the "real_time_preprocessing" notebook (using the table method, not actual real time)
# MAGIC   - df_preprocessed has reduced play time to gameplay windows according to possession change, shot attempt, rebound, gc stopped (wcTime discontinuous).  And these gameplay windows are *uniquely* defined by the (gameId, dividerId) doublet
# MAGIC   - In order to find "similarity scores" you must have a "seed play", i.e. "I want the top 100 plays (unique (gameId, dividerId)) that are most like *this* play (seed play: (gameId, dividerId))"
# MAGIC   - Event Chain logic is a separate model that will apply downstream of this ranking pipeline, and combine the results of itself with the rankings of this heatmaps pipline
# MAGIC 
# MAGIC   
# MAGIC ### Logical Steps
# MAGIC 
# MAGIC The process/pipeline from df_preprocessed to similarity score ranked list, can be thought of as *five* logical steps across *two* notebooks.  These steps are:
# MAGIC   1) pull in the tracking data with (preprocessed) features assigned to dividerIds, which is the output of real_time_preprocessing, we call it "df_preprocessed"
# MAGIC   2) organize the tracking data into counts of position (and whatever other heatmaps created in the future) into a table per type of heatmap. (e.g. one table for position heatmaps, one table for velocity heatmaps...)
# MAGIC   3) pull in all above tables, aggegate the cours, and put each heatmap court as a column in one dataframe, and split by test and train dataframes, and write out test and train dataframes
# MAGIC   4) Train the VAE and organize results, plot visuals
# MAGIC   5) Write out the results of searched given a set of seed plays
# MAGIC   
# MAGIC   
# MAGIC ### Notebooks
# MAGIC 
# MAGIC The logical steps above are to be found within two notbooks.
# MAGIC   - A) Generate Heatmaps: has logical steps 1) and 2) from the above list.  Notebook: https://adb-8454710615903158.18.azuredatabricks.net/?o=8454710615903158#notebook/3929691496281079
# MAGIC   - B) Traing VAE: has logical steps 3), 4), and 5). Notebook: https://adb-8454710615903158.18.azuredatabricks.net/?o=8454710615903158#notebook/2707405312756802

# COMMAND ----------

# MAGIC %md
# MAGIC ### Heatmap Generation Notebook Details
# MAGIC Notebook: https://adb-8454710615903158.18.azuredatabricks.net/?o=8454710615903158#notebook/3929691496281079
