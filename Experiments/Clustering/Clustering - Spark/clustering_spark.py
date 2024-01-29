from pyspark.sql import SparkSession
from pyspark.sql.functions import concat,col

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import os
import sys
import time

import threading

num_threads = 3
num_graphs = 3

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ["SPARK_HOME"] = "/opt/spark/"

spark = SparkSession \
     .builder \
     .master("spark://master:7077").getOrCreate()

print("spark session created")



def clustering(i):
	# Read dataframe
	data = spark.read.option("header", "true").option("inferSchema", "true").parquet(f"hdfs://master:9000/user/hdoop/Clustering/cluster{i}")

	num_features = len(data.columns) - 2

	# Assemble columns to one
	assembler =\
	    VectorAssembler(inputCols=[f"feat_{i+1}" for i in range(num_features)], outputCol="assembledFeatures")

	# Scale data
	scaler =\
	    StandardScaler(inputCol="assembledFeatures",outputCol="scaledFeatures")

	tr_assembler = assembler.transform(data)
	tr_scaler = scaler.fit(tr_assembler)
	tr_scaler = tr_scaler.transform(tr_assembler)

	KMC = KMeans(k=5,featuresCol="scaledFeatures",predictionCol="prediction",maxIter=2)

	# Create model and fit data
	model = KMC.fit(tr_scaler)
	transformed = model.transform(tr_scaler)


	centers = model.clusterCenters() 
	print("Cluster Centers: ") 
	for center in centers: 
	    print(center)


	"""
	# Evaluate clustering by computing Silhouette score
	ClusterEvaluator = ClusteringEvaluator(predictionCol="prediction",featuresCol="scaledFeatures")

	silhouette = ClusterEvaluator.evaluate(transformed)
	"""
	
#print(f"Silhouette with squared euclidean distance and 5 clusters = " + str(silhouette))



# Get start time
start = time.time()


threads = [threading.Thread(target=clustering,args=[j+1]) for j in range(num_threads)]

for j in range(num_threads):
	threads[j].start()
	
for j in range(num_threads):
	threads[j].join()

# Get end time
end = time.time()
# Write stats about time, workers, memory and cores to file
num_workers = sys.argv[1]
worker_info = []
for i in range(2,len(sys.argv)):
	worker_info.append(sys.argv[i].split(","))

out = open("spark_stats.txt","a")

out.write(f"Total Time: {end-start}\n")
out.write(f"Total Workers: {num_workers}\n")

for i,worker in enumerate(worker_info):
	out.write(f"   Worker {i+1}:\n")
	out.write(f"      Memory: {worker[0]}G\n")
	out.write(f"      CPU Cores: {worker[1]}\n")

out.write("\n")
