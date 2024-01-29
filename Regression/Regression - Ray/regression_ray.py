import ray

import os

import time

import sys
import gc


import pandas as pd

import numpy as np



from joblib import parallel_backend

from ray.util.joblib import register_ray



from sklearn.linear_model import Ridge

from sklearn.impute import SimpleImputer



from sklearn.metrics import mean_squared_error

#from xgboost_ray import RayDMatrix, RayParams, RayXGBRegressor,predict



register_ray()



#os.environ['CLASSPATH'] = '/usr/local/hadoop/bin/hdfs classpath --glob'

os.environ['HADOOP_HOME'] = '/usr/local/hadoop/'

os.environ['ARROW_LIBHDFS_DIR'] = '/usr/local/hadoop/lib/native/'



@ray.remote

def regression_distributed(X_train, X_test, y_train, y_test):



	imputer = SimpleImputer(strategy='mean')



	X_train_imp = imputer.fit_transform(X_train)

	X_test_imp = imputer.fit_transform(X_test)



	ridge = Ridge(max_iter=10,tol=0.0001)


	gc.collect()

	print("Training...")

	ridge.fit(X_train_imp,y_train)



	print("Predicting...")

	preds = ridge.predict(X_test_imp)



	return mean_squared_error(preds,y_test)



ray.init()



# Get start time

start = time.time()



with parallel_backend('ray'):

	df_reg_train = ray.data.read_parquet("hdfs://master:9000/user/hdoop/Regression/train/").to_pandas()



	num_features = len(df_reg_train.columns)-2



	X_train = df_reg_train[[f"feat_{i+1}" for i in range(num_features)]]

	y_train = df_reg_train["label"]



	df_reg_test = ray.data.read_parquet("hdfs://master:9000/user/hdoop/Regression/test/").to_pandas()



	X_test = df_reg_test[[f"feat_{i+1}" for i in range(num_features)]]

	y_test = df_reg_test["label"]



	ray.data.DataContext.get_current().execution_options.verbose_progress = True



	rmse = ray.get(regression_distributed.options(num_cpus=0,scheduling_strategy="SPREAD").remote(X_train,X_test,y_train,y_test))

#print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)



# Get end time

end = time.time()



# Write stats about time, workers, memory and cores to file

num_workers = sys.argv[1]

worker_info = []



for i in range(2,len(sys.argv)):

	worker_info.append(sys.argv[i].split(","))



out = open("ray_stats.txt","a")


out.write(f"Root Mean Squared Error (RMSE) on test data = {rmse}\n")
out.write(f"Total Time: {end-start}\n")

out.write(f"Total Workers: {num_workers}\n")



for i,worker in enumerate(worker_info):

	out.write(f"   Worker {i+1}:\n")

	out.write(f"      Memory: {worker[0]}G\n")

	out.write(f"      CPU Cores: {worker[1]}\n")



out.write("\n")

