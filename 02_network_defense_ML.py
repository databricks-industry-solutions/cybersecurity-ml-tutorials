# Databricks notebook source
# MAGIC %run ./01_network_defense_setup

# COMMAND ----------

# MAGIC %md #Theory of Network Defense
# MAGIC Networks are complex systems. From a defensive perspective, networks may be open to a broad range of attack surfaces and threat vectors. This means that network defenders must engage with attackers on multiple fronts. Importantly, we must make not make assumptions about individual components of the system.
# MAGIC 
# MAGIC 
# MAGIC ### Machine Learning and Network Security
# MAGIC A primary strength of machine learning is pattern mining, or identifying rules that describe specific patterns within a dataset. Network traffic is strictly governed by a set of protocols that result in structures and patterns in the data. We can use pattern mining techniques to identify malicious activity by looking for patterns and drawing correlations in the data. This is especially useful for attacks that rely on volume and/or iteration (e.g. network scanning and DoS attacks).
# MAGIC 
# MAGIC **Scenario:**
# MAGIC Our task is to devise a general classifier that categorizes each sample as one of five classes:
# MAGIC 1. benign
# MAGIC 2. dos - Denial of service
# MAGIC 3. r2l - Unauthorized accesses from remote servers
# MAGIC 4. u2r - Privilege escalation attempts
# MAGIC 5. probe - Brute force probing attacks
# MAGIC 
# MAGIC **Reference:** Demo is based on these [resources](https://github.com/oreilly-mlsec/book-resources/tree/master/chapter5)
# MAGIC 
# MAGIC **Databricks Capabilities:**
# MAGIC Databricks Machine Learning is an integrated end-to-end machine learning environment incorporating managed services for experiment tracking, model training, feature development and management, and feature and model serving. The diagram shows how the capabilities of Databricks map to the steps of the model development and deployment process.
# MAGIC 
# MAGIC <div><img width="800" src="https://docs.databricks.com/_images/ml-diagram-model-development-deployment.png"/></div>

# COMMAND ----------

# MAGIC %md ## Why Databricks?
# MAGIC 
# MAGIC *The hardest thing about security analytics aren’t the analytics.*
# MAGIC 
# MAGIC Analyzing large scale DNS traffic logs is complicated - challenges include:
# MAGIC 
# MAGIC - **Complexity:** DNS server data is everywhere. Cloud, hybrid, and multi-cloud deployments make it challenging to collect the data, have a single data store and run analytics consistently across the entire deployment.
# MAGIC - **Tech limitations:** Legacy SIEM and log aggregation solutions can’t scale to cloud data volumes for storage, analytics or ML/AI workloads.
# MAGIC - **Cost:** SIEMs or log aggregation systems charge by volume of data ingest. With so much data, licensing and hardware requirements make DNS analytics cost prohibitive. And moving data from one cloud service provider to another is also costly and time consuming. The hardware pre-commit in the cloud or the expense of physical hardware on-prem are all deterrents for security teams.
# MAGIC 
# MAGIC 
# MAGIC Databricks provides a real-time data analytics platform that can handle cloud-scale, analyze data wherever it is, natively support streaming and batch analytics and provide collaborative, content development capabilities.
# MAGIC 
# MAGIC 
# MAGIC <div><img width="800" src="https://www.databricks.com/wp-content/uploads/2020/10/blog-detecting-criminals-1.png"/></div>

# COMMAND ----------

# MAGIC %md # 1/ Data Exploration
# MAGIC 
# MAGIC The dataset that we will use is the **NSL-KDD dataset**, which is an improvemenet to the original 1999 KDD Cup dataset created for the DARPA Intrusion Detection Evaluation Program. This data was collected over nine weeks and consists of raw tcpdump traffic that simulates the environment of a typical USAF LAN. Some network attacks were deliberately carried out during the recording period. There are 24 types of attacks available in our training set.
# MAGIC 
# MAGIC Data Source: https://www.unb.ca/cic/datasets/nsl.html
# MAGIC </br>
# MAGIC Paper: https://www.ecb.torontomu.ca/~bagheri/papers/cisda.pdf

# COMMAND ----------

# MAGIC %fs ls /tmp/cyber_ml

# COMMAND ----------

# DBTITLE 1,Load Training & Test Data
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

trainingdf = spark.read.table(f"{getParam('db')}.training_data")
testingdf = spark.read.table(f"{getParam('db')}.testing_data")
header_names = testingdf.columns

# Differentiating between nominal, binary, and numeric features
column_names = np.array(header_names)

nominal_idx = [1, 2, 3]
binary_idx = [6, 11, 13, 14, 20, 21]
numeric_idx = list(set(range(41)).difference(nominal_idx).difference(binary_idx))

nominal_cols = column_names[nominal_idx].tolist()
binary_cols = column_names[binary_idx].tolist()
numeric_cols = column_names[numeric_idx].tolist()

display(testingdf)

# COMMAND ----------

# DBTITLE 1,Category Distribution
# training_attack_types.txt maps each of the different attacks to 1 of 4 categories
# the data is obtained from http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types

categories = ['benign', 'dos', 'u2r', 'r2l', 'probe']

attack_mapping = {
 'normal': 'benign',
 'apache2': 'dos',
 'back': 'dos',
 'mailbomb': 'dos',
 'processtable': 'dos',
 'snmpgetattack': 'dos',
 'teardrop': 'dos',
 'smurf': 'dos',
 'land': 'dos',
 'neptune': 'dos',
 'pod': 'dos',
 'udpstorm': 'dos',
 'ps': 'u2r',
 'buffer_overflow': 'u2r',
 'perl': 'u2r',
 'rootkit': 'u2r',
 'loadmodule': 'u2r',
 'xterm': 'u2r',
 'sqlattack': 'u2r',
 'httptunnel': 'u2r',
 'ftp_write': 'r2l',
 'guess_passwd': 'r2l',
 'snmpguess': 'r2l',
 'imap': 'r2l',
 'spy': 'r2l',
 'warezclient': 'r2l',
 'warezmaster': 'r2l',
 'multihop': 'r2l',
 'phf': 'r2l',
 'named': 'r2l',
 'sendmail': 'r2l',
 'xlock': 'r2l',
 'xsnoop': 'r2l',
 'worm': 'probe',
 'nmap': 'probe',
 'ipsweep': 'probe',
 'portsweep': 'probe',
 'satan': 'probe',
 'mscan': 'probe',
 'saint': 'probe'}

display(attack_mapping)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze train and test sets
# MAGIC It is important to consider class distribution within the trainng and test data sets because it can helpful to get a general idea of what is expected. Since we have access to the test data in this example, we will consume the data to get its distribution.

# COMMAND ----------

train_df = trainingdf.toPandas()
train_df['attack_category'] = train_df['attack_type'].map(lambda x: attack_mapping[x])
train_df.drop(['success_pred'], axis=1, inplace=True)
    
test_df = testingdf.toPandas()
test_df['attack_category'] = test_df['attack_type'].map(lambda x: attack_mapping[x])
test_df.drop(['success_pred'], axis=1, inplace=True)

# COMMAND ----------

# DBTITLE 1,Training Data Class Distribution (20-category breakdown)
train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['attack_category'].value_counts()

test_attack_types = test_df['attack_type'].value_counts()
test_attack_cats = test_df['attack_category'].value_counts()

train_attack_types.plot(kind='barh', figsize=(12,6), fontsize=12)

# COMMAND ----------

# DBTITLE 1,5-category breakdown
train_attack_cats.plot(kind='barh', figsize=(10,5), fontsize=15)

# COMMAND ----------

# Binary features - all of these features should have a min of 0.0 and a max of 1.0
train_df[binary_cols].describe().transpose()

# COMMAND ----------

# Note the su_attempted column has a max value of 2.0. Why is that?
train_df.groupby(['su_attempted']).size()

# COMMAND ----------

# Let's fix this and assume that su_attempted=2 really means su_attempted=0
train_df['su_attempted'].replace(2, 0, inplace=True)
test_df['su_attempted'].replace(2, 0, inplace=True)
train_df.groupby(['su_attempted']).size()

# COMMAND ----------

# We also notice that the num_outbound_cmds column only takes on one value
train_df.groupby(['num_outbound_cmds']).size()

# This is not a useful feature - let's drop it from the dataset
train_df.drop('num_outbound_cmds', axis = 1, inplace=True)
test_df.drop('num_outbound_cmds', axis = 1, inplace=True)
numeric_cols.remove('num_outbound_cmds')

# COMMAND ----------

# MAGIC %md
# MAGIC # 2/ Data Preparation

# COMMAND ----------

# DBTITLE 1,Split Test & Training into Data & Lables
train_Y = train_df['attack_category']
train_x_raw = train_df.drop(['attack_category','attack_type'], axis=1)

test_Y = test_df['attack_category']
test_x_raw = test_df.drop(['attack_category','attack_type'], axis=1)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Encoding Categorial Variables
# MAGIC We'll use pandas.get_dummies() to convert the nominal variables into dummy variables.  This is necessary because there might be some symbolic variables that appear in one dataset and not the other, and separately generating dummy variables for them would result in inconsistencies in the columns of both datasets.

# COMMAND ----------

combined_df_raw = pd.concat([train_x_raw, test_x_raw])
combined_df = pd.get_dummies(combined_df_raw, columns=nominal_cols, drop_first=True)

train_x = combined_df[:len(train_x_raw)]
test_x = combined_df[len(train_x_raw):]

# Store dummy variable feature names
dummy_variables = list(set(train_x) - set(combined_df_raw))

# COMMAND ----------

# MAGIC %md ### Standardization
# MAGIC The distributions of the feautres vary widely, and this may affect our results if we use any distance-based methods for classification. For example, the mean of src_bytes is larger than the mean of num_failed_logins by seven orders of magnitude. Without performing feature value normalization, the src_bytes feature would dominate.

# COMMAND ----------

train_x.describe()

# COMMAND ----------

# DBTITLE 1,Descriptive Statistics
# Example statistics for the 'duration' feature before scaling
train_x['duration'].describe()

# COMMAND ----------

# Let's experiment with StandardScaler on the 'duration' feature. 
# This will standardize the feature by removing the mean and scaling to unit variance.
from sklearn.preprocessing import StandardScaler

durations = train_x['duration'].values.reshape(-1, 1)
standard_scaler = StandardScaler().fit(durations)
scaled_durations = standard_scaler.transform(durations)
pd.Series(scaled_durations.flatten()).describe()

# COMMAND ----------

# MAGIC %md ^^^ Now we see that the series has been scaled to have a mean of close to 0 with a standard deviation of close to 1

# COMMAND ----------

# DBTITLE 1,Apply StandardScaler to All Numeric Columns
# Let's apply to all the numeric columns

standard_scaler = StandardScaler().fit(train_x[numeric_cols])
train_x[numeric_cols] = standard_scaler.transform(train_x[numeric_cols])
test_x[numeric_cols] = standard_scaler.transform(test_x[numeric_cols])

# COMMAND ----------

# MAGIC %md #3/ Classification
# MAGIC At this point we have a five-class classification problem in which each sample belongs to one of five classes. There are many different suitable  algorithms for a problem like this, and many different ways to approach the problem of multiclass classification.
# MAGIC 
# MAGIC In general, here are some questions you can ask yourself when selecting an ML algorithm:
# MAGIC - What is the size of your training set?
# MAGIC - Are you predicting a category or quantitative value?
# MAGIC - Do you have labeled data? How much?
# MAGIC - How much time and resources do you have to train?
# MAGIC - How much time and resources do you have to make a prediction?

# COMMAND ----------

# MAGIC %md ##Supervised Learning
# MAGIC 
# MAGIC Given that we have 126,000 labeled samples, supervised training is a great place to start.

# COMMAND ----------

# DBTITLE 1,Supervised Learning - DecisionTree
import mlflow

# 5-class classification version
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn import tree
from mlflow.models.signature import infer_signature

# Enable autolog()
mlflow.sklearn.autolog(silent="true")
 
# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run(run_name="classification_decisiontree"):
  classifier = DecisionTreeClassifier(random_state=17)
  classifier.fit(train_x, train_Y)

  pred_y = classifier.predict(test_x)
  results = confusion_matrix(test_Y, pred_y)
  error = zero_one_loss(test_Y, pred_y)
  fig = tree.plot_tree(classifier, max_depth=4)
  
  ###########################################
  #      Log extra parameters to MLFlow      #
  ###########################################
  mlflow.set_tag("project", "network_classification")
  mlflow.log_metric("depth", classifier.get_depth())
  mlflow.log_metric("error", error)
  
  #signature = infer_signature(test_x, classifier.predict(test_x))
  mlflow.sklearn.log_model(classifier, "decisiontreeclf")  

print(results)
print(error)

# COMMAND ----------

# MAGIC %md ### Not Bad!
# MAGIC With just a little code and no tuning we have a 76.2% classification accuracy

# COMMAND ----------

# MAGIC %md #### Register the model in MLflow Model Registry
# MAGIC By registering this model in Model Registry, you can easily reference the model from anywhere within Databricks.
# MAGIC 
# MAGIC The following section shows how to do this programmatically, but you can also register a model using the UI.

# COMMAND ----------

# Get the latest model from the registry
run_id = mlflow.search_runs()['run_id'][0]
model_registered = mlflow.register_model("runs:/"+run_id+"/decisiontreeclf", "network_classification")

# COMMAND ----------

# MAGIC %md #4/ Deploying & using our model in production
# MAGIC 
# MAGIC Now that our model is in our MLFlow registry, we can start to use it in a production pipeline.

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(name = "network_classification", version = model_registered.version, stage = "Production", archive_existing_versions=True)

print("registering model version "+run_id+" as production model")

# COMMAND ----------

from pyspark.sql.functions import struct, col
logged_model = 'runs:/d6c7d1c60f054ec1829c184b27bac5c4/decisiontreeclf'

df = spark.createDataFrame(test_x)

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='string')

# Predict on a Spark DataFrame.
display(df.withColumn('predictions', loaded_model(struct(*map(col, df.columns)))))

# COMMAND ----------

# DBTITLE 1,KNeighbors
from sklearn.neighbors import KNeighborsClassifier

with mlflow.start_run(run_name="classification_KNN"):
  classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
  classifier.fit(train_x, train_Y)

  pred_y = classifier.predict(test_x)

  results = confusion_matrix(test_Y, pred_y)
  error = zero_one_loss(test_Y, pred_y)
  
  ###########################################
  #      Log extra parameter to MLFlow      #
  ###########################################
  mlflow.set_tag("project", "network_classification")  

print(results)
print(error)

# COMMAND ----------

# DBTITLE 1,LinearSVC
from sklearn.svm import LinearSVC

# Linear Support Vector Machine
with mlflow.start_run(run_name="classification_LinearSVC"):
  classifier = LinearSVC()
  classifier.fit(train_x, train_Y)

  pred_y = classifier.predict(test_x)

  results = confusion_matrix(test_Y, pred_y)
  error = zero_one_loss(test_Y, pred_y)

  ###########################################
  #      Log extra parameters to MLFlow      #
  ###########################################
  mlflow.set_tag("project", "network_classification")
  mlflow.log_metric("error", error)
  mlflow.sklearn.log_model(classifier, "linearsvc_clf")  
  
print(results)
print(error)

# COMMAND ----------

# MAGIC %md ##Unsupervised Learning
# MAGIC What if you have no labeled data at all? Labeled data can be extremely difficult to generate (especially in the security space). In the classification space, the dominant class of methods for unsupervised learning is clustering.

# COMMAND ----------

# DBTITLE 1,Visualize Numeric Columns - Principal Component Analysis
# First, let's visualize the dataset (only numeric columns)

from sklearn.decomposition import PCA

with mlflow.start_run(run_name="classification_PCA"):
  # Use PCA to reduce dimensionality so we can visualize the dataset on a 2d plot
  pca = PCA(n_components=2)
  train_x_pca_cont = pca.fit_transform(train_x[numeric_cols])

  plt.figure(figsize=(15,10))
  colors = ['navy', 'turquoise', 'darkorange', 'red', 'purple']

  for color, cat in zip(colors, categories):
      plt.scatter(train_x_pca_cont[train_Y==cat, 0], train_x_pca_cont[train_Y==cat, 1],
                  color=color, alpha=.8, lw=2, label=cat)
  plt.legend(loc='best', shadow=False, scatterpoints=1)

  plt.show()

# COMMAND ----------

# MAGIC %md #### PCA (Whomp whomp...)
# MAGIC 
# MAGIC This dataset does not appear to be very suitable for clustering. The probe, dos, and r2l samples are scattered unpredictably and do not form any strong clusters. Because there are few u2r samples, it is difficult to observe this class on the plot. Only benign samples seem to form a strong cluster.

# COMMAND ----------

# DBTITLE 1,Let's Try with K-Means
# Apply k-means (k=5, only using numeric cols) + PCA + plot

from sklearn.cluster import KMeans

with mlflow.start_run(run_name="classification_KMeans"):
  # Fit the training data to a k-means clustering estimator model
  kmeans = KMeans(n_clusters=5, random_state=17).fit(train_x[numeric_cols])

  # Retrieve the labels assigned to each training sample
  kmeans_y = kmeans.labels_

  # Plot in 2d with train_x_pca_cont
  plt.figure(figsize=(15,10))
  colors = ['navy', 'turquoise', 'darkorange', 'red', 'purple']

  for color, cat in zip(colors, range(5)):
      plt.scatter(train_x_pca_cont[kmeans_y==cat, 0],
                  train_x_pca_cont[kmeans_y==cat, 1],
                  color=color, alpha=.8, lw=2, label=cat)
  plt.legend(loc='best', shadow=False, scatterpoints=1)
  plt.show()

# COMMAND ----------

# MAGIC %md #### K-Means
# MAGIC 
# MAGIC Immediately we see differences! The algorithm performs well in grouping certain sections of the data, but it fails to group together clusters of the dos label and wrongly classifies a large section of "benign" traffic as attack traffic. Tuning and experimeing with the value of k can help, but we can see that there might be a fundamental problem with using only clustering methods to classify network attacks in this data.
