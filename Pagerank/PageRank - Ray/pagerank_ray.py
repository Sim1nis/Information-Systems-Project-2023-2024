import ray
import os
import time
import sys

import joblib

import pandas as pd
import numpy as np
import networkx as nx
from pyarrow import csv

import ray
from ray.util.joblib import register_ray



@ray.remote
class myGraph:
    def __init__(self,edges,alpha):
        self.graph = nx.from_pandas_edgelist(edges,'src','dst')
        self.x = (1-alpha) * np.ones(self.graph.number_of_nodes()) 
        self.x_next = (1-alpha) * np.ones(self.graph.number_of_nodes()) 
        self.alpha = alpha

    def subtract(self,n):
        return abs(self.x_next[n] - self.x[n])

    def compute_x(self,n):
            self.x_next[n] = self.alpha * sum([self.x[m]/self.graph.degree(m) for m in self.graph.neighbors(n)]) + (1-self.alpha)

    def get_nodes(self):
         return self.graph.nodes()
    
    def get_x(self):
        return self.x
    
    def set_x(self):
        self.x = self.x_next.copy()

# Get start time

register_ray()


ray.init()

start=time.time()

#with joblib.parallel_backend('ray'):
alpha = 0.85
parse_options = csv.ParseOptions(delimiter=" ")

    #edges = ray.data.read_text("hdfs://master:9000/user/hdoop/custom_graph/email-Eu-core.txt").to_pandas()
with joblib.parallel_backend('ray'):
     edges = ray.data.read_csv("hdfs://master:9000/user/hdoop/Graph/edges/graph_edges_1.txt",parse_options=csv.ParseOptions(delimiter=" ")).to_pandas().drop_duplicates()

edges.rename(columns={edges.columns[0]:"src",edges.columns[1]:"dst"},inplace=True)

    #edges["src"] = edges.apply(lambda row: int(row["text"].split(" ")[0]),axis=1)
    #edges["dst"] = edges.apply(lambda row: int(row["text"].split(" ")[1]),axis=1)

    #edges.drop("text",axis=1,inplace=True)

#    vertices = ray.data.read_text("hdfs://master:9000/user/hdoop/custom_graph/email-Eu-vertices.txt").to_pandas()

"""
    edges =edges.map(lambda row: {'src':int(row["text"].split(" ")[0]),'dst':int(row["text"].split(" ")[1])})

    vertices = vertices.map(lambda row: {'id': int(row["text"])})
    vertices = vertices.map(lambda vrow: {'neigh': edges.filter(lambda erow: erow['src'] == 1,concurrency=1000).to_pandas()["dst"].values.tolist()})
    
    print(vertices.take(10))
"""

print("Building graph...")

G= myGraph.remote(edges,alpha)

print("Graph built!")

print("Executing PageRank...")
nodes = ray.get(G.get_nodes.remote())
while True:
        with joblib.parallel_backend('ray'):
             compute_deg = [G.compute_x.remote(n) for n in nodes]
        print("yay,got here")
        ray.get(compute_deg)
        err_arr = ray.get(([G.subtract.remote(n) for n in nodes]))
        if sum(err_arr) < 1.0e-2:
            break
        ray.get(G.set_x.remote())
        print(sum(err_arr))

print("Error:",sum(err_arr))

res = ray.get(G.get_x.remote())
res = list(res)
total = sum(res)
print(res[121]) 
res.sort(reverse=True)
print([res[i]*len(res)/total for i in range(10)])

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
