# Databricks notebook source
# MAGIC %pip install graphviz
# MAGIC %pip install pydotplus

# COMMAND ----------

# MAGIC %run ./config/notebook_config

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC mkdir /dbfs/tmp/cyber_ml
# MAGIC cd /dbfs/tmp/cyber_ml
# MAGIC pwd
# MAGIC echo "Removing all files"
# MAGIC rm -rf *
# MAGIC 
# MAGIC fname="NSL-KDD.zip"
# MAGIC dlpath="https://raw.githubusercontent.com/lipyeow-lim/security-datasets01/main/nsl-kdd-1999/$fname"
# MAGIC wget $dlpath
# MAGIC unzip $fname
# MAGIC 
# MAGIC ls -lR

# COMMAND ----------

sql_list = [ f"""
CREATE SCHEMA IF NOT EXISTS {getParam('db')} COMMENT 'This is a network cyber ml demo schema'""",
f"""USE SCHEMA {getParam('db')}""" ]

for s in sql_list:
  print(s)
  spark.sql(s)

# COMMAND ----------

dataset_root = getParam("download_path")
train_file = os.path.join(dataset_root, 'KDDTrain+.txt')
test_file = os.path.join(dataset_root, 'KDDTest+.txt')


# COMMAND ----------

header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'success_pred']

# COMMAND ----------

testdf = (spark.read
  .format("csv")
  .option("mode", "PERMISSIVE")
  .option("inferSchema", "true")
  .load(test_file)
)

testdf = testdf.toDF(*header_names)
testdf.write.mode("overwrite").saveAsTable("testing_data")

# COMMAND ----------

trainingdf = (spark.read
  .format("csv")
  .option("mode", "PERMISSIVE")
  .option("inferSchema", "true")
  .load(train_file)
)

trainingdf = trainingdf.toDF(*header_names)
trainingdf.write.mode("overwrite").saveAsTable("training_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select *
# MAGIC from testing_data

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select *
# MAGIC from training_data
