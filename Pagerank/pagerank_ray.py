import ray

import os

import time



import pandas as pd

import numpy as np

import networkx as nx



from joblib import parallel_backend

from ray.util.joblib import register_ray



os.environ['HADOOP_HOME'] = '/usr/local/hadoop/'

os.environ['ARROW_LIBHDFS_DIR'] = '/usr/local/hadoop/lib/native/'
register_ray()


ray.init()



# Get start time
with parallel_backend("ray"):
	start = time.time()



	edges = ray.data.read_text("hdfs://master:9000/user/hdoop/Graph/edges/") #for i in range(1,45)]).take_all()

	vertices = ray.data.read_text("hdfs://master:9000/user/hdoop/Graph/vertices/")



	d = {"source": [e["text"].split()[0] for e in edges] + [e["text"].split()[1] for e in edges], "target":[e["text"].split()[1] for e in edges] + [e["text"].split()[0] for e in edges]}

	df_edge = pd.DataFrame(data=d)



	graph = nx.from_pandas_edgelist(df_edge,"source","target")

	res = nx.pagerank(graph,alpha=0.15,max_iter=10)



	pagerank_df = pd.DataFrame({"vertice":res.keys(),"score":res.values()})

	pagerank_df["score"] *= 1005



	print(pagerank_df.sort_values(by='score',ascending=False))

# Get end time

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
