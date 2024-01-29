import ray
import os
import time
import sys

import pandas as pd
import numpy as np

from joblib import parallel_backend
from ray.util.joblib import register_ray
from ray.data.preprocessors import SimpleImputer

from sklearn.metrics import accuracy_score
from xgboost_ray import RayDMatrix, RayParams, RayXGBClassifier,predict

register_ray()

#os.environ['CLASSPATH'] = '/usr/local/hadoop/bin/hdfs classpath --glob'
os.environ['HADOOP_HOME'] = '/usr/local/hadoop/'
os.environ['ARROW_LIBHDFS_DIR'] = '/usr/local/hadoop/lib/native/'

ray.init()

# Get start time
start = time.time()

with parallel_backend('ray'):
	X_train = ray.data.read_parquet("hdfs://master:9000/user/hdoop/Classification/train/")

	#num_features = len(ds_reg_train.columns())-2

	#X_train = ds_reg_train.select_columns([f"feat_{i+1}" for i in range(num_features)])
	#y_train = ds_reg_train.select_columns("label")

	X_test = ray.data.read_parquet("hdfs://master:9000/user/hdoop/Classification/test/")

	#X_test = ds_reg_test.select_columns([f"feat_{i+1}" for i in range(num_features)])
	#y_test = ds_reg_test.select_columns("label")

	num_features = len(X_train.columns())-1

	imputer = SimpleImputer(columns=[f"feat_{i+1}" for i in range(num_features)], strategy='mean')

	X_train_imp = imputer.fit_transform(X_train)
	X_test_imp = imputer.fit_transform(X_test)

	X_train_xgb = RayDMatrix(X_train_imp,label='label')

	clf = RayXGBClassifier(objective='binary:logistic', num_class=2, n_estimators=100,learning_rate=0.01,n_jobs=3)

	print("Training...")
	clf.fit(X_train_xgb,None,ray_params=RayParams(num_actors=3,cpus_per_actor=2))

	print("Predicting...")
	preds = clf.predict(X_test_imp.select_columns([f"feat_{i+1}" for i in range(num_features)]).to_pandas(),ray_params=RayParams(num_actors=3,cpus_per_actor=2))

	acc = accuracy_score(preds,X_test_imp.to_pandas()["label"].to_numpy())

	print("Accuracy on test data = %g" % acc)

# Get end time
end = time.time()

# Write stats about time, workers, memory and cores to file
num_workers = sys.argv[1]
worker_info = []

for i in range(2,len(sys.argv)):
	worker_info.append(sys.argv[i].split(","))

out = open("ray_stats.txt","a")

out.write(f"Accuracy on test data = {acc}\n")
out.write(f"Total Time: {end-start}\n")
out.write(f"Total Workers: {num_workers}\n")

for i,worker in enumerate(worker_info):
	out.write(f"   Worker {i+1}:\n")
	out.write(f"      Memory: {worker[0]}G\n")
	out.write(f"      CPU Cores: {worker[1]}\n")

out.write("\n")

