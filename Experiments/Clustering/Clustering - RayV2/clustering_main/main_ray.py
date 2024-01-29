import _k_means_ray
import _k_means_elkan
import _k_means_fast
import _k_means_spark
import remote_loader

import os
import sys
import time
import getopt
import numpy as np
import ray
from numpy import array

import joblib
from sklearn.cluster import KMeans
from ray.util.joblib import register_ray

from pyspark import SparkContext
import pyspark.mllib.clustering

@ray.remote(scheduling_strategy="SPREAD")
class Pipeline:
    def __init__(self, actor_id, cluster_k=5, iteration=10):
        self.cluster_k = cluster_k
        self.iteration = iteration
        self.center = None
        self.actor_id = actor_id
        self.df = ray.data.read_parquet(f"hdfs://master:9000/user/hdoop/Clustering/cluster{self.actor_id+1}").drop_columns("cluster").to_pandas()
    def cluster_ray(self, batch_num, init_method="k-means++", assign_method="full" ,task_num=2):

        # split data
        batches = _k_means_ray.splitData(self.df, num=batch_num)
        print(batches)
        # init center
        center = _k_means_ray._initK(
            self.df, self.cluster_k, method=init_method)
        #print(center)
        n = center.shape[0]  # n center points
        distMatrix = np.empty(shape=(n, n))
        _k_means_fast.createDistMatrix(center, distMatrix)

        # init ray
        mappers = [_k_means_ray.KMeansMapper.remote(
            mini_batch.values, k=self.cluster_k) for mini_batch in batches[0]]
        reducers = [_k_means_ray.KMeansReducer.remote(
            i, *mappers) for i in range(self.cluster_k)]
        start = time.time()
        cost = 0

        for i in range(self.iteration):
            # broadcast center point
            for mapper in mappers:
                mapper.broadcastCentroid.remote(center)
                if(assign_method == "elkan" or assign_method == "mega_elkan"):
                    mapper.broadcastDistMatrix.remote(distMatrix)

            # map function
            for mapper in mappers:
                mapper.assignCluster.remote(
                    method=assign_method, task_num=task_num)

            newCenter, cost = _k_means_ray.createNewCluster(reducers)
            changed, cost_1 = _k_means_ray.isUpdateCluster(
                newCenter, center)  # update
            if (not changed):
                break
            else:
                center = newCenter
                if(assign_method == "elkan" or assign_method == "mega_elkan"):
                    _k_means_fast.createDistMatrix(center, distMatrix)
                print(str(i) + " iteration, cost: " + str(cost))

        # print(center)
        end = time.time()
        print(center)
        self.center = center
        print('execution time: ' + str(end-start) + 's, cost: ' + str(cost))

    def cluster_sklearn(self, init_method="k-means++", assign_method="elkan", n_jobs=1):
        start = time.time()

        ml = KMeans(n_clusters=self.cluster_k,  init=init_method, verbose=1,
                    n_jobs=n_jobs, max_iter=self.iteration, algorithm=assign_method)

        ml.fit(self.df)

        end = time.time()
        center = ml.cluster_centers_
        print(center)
        self.center = center

        print('execution time: ' + str(end-start) + 's')

    def cluster_spark(self, output_file='test.txt', init_method="random", epsilon=1e-4):
        start = time.time()
        output_name = './data/' + output_file
        self.dataprocessor.saveData(self.df, output_file)
        sc = SparkContext(appName="KmeansSpark")
        data = sc.textFile(output_name)
        parsedData = data.map(lambda line: array(
            [float(x) for x in line.split('\t')]))

        # Build the model (cluster the data)
        clusters = pyspark.mllib.clustering.KMeans.train(parsedData, k=self.cluster_k, maxIterations=self.iteration,
                                                         initializationMode=init_method, epsilon=epsilon)
        end = time.time()
        center = np.array(clusters.centers)
        print(center)
        self.center = center
        print('execution time: ' + str(end-start) + 's')
    

number_of_clusters = 5
number_of_iteration = 5
number_of_mappers = 5
number_of_tasks = 2

actor_num = 3

ray.init()

#remote_loader.remote_import(['./_k_means_ray','./_k_means_spark','./_k_means_elkan'])

start = time.time()

worker = [Pipeline.remote(actor_id=j, cluster_k = number_of_clusters,iteration=number_of_iteration) for j in range(actor_num)]
ray.get([worker[j].cluster_ray.remote(batch_num=number_of_mappers, init_method="random", assign_method="full", task_num=number_of_tasks) for j in range(actor_num)])

end = time.time()

# Write stats about time, workers, memory and cores to file
num_workers = sys.argv[1]
worker_info = []

for i in range(2,len(sys.argv)):
	worker_info.append(sys.argv[i].split(","))

out = open("ray_stats.txt","a")

out.write(f"Total Time: {end-start}\n")
out.write(f"Total Workers: {num_workers}\n")

for i,worker in enumerate(worker_info):
	out.write(f"   Worker {i+1}:\n")
	out.write(f"      Memory: {worker[0]}G\n")
	out.write(f"      CPU Cores: {worker[1]}\n")

out.write("\n")
