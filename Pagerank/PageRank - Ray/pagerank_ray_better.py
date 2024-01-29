import ray
import os
import time
import sys
import gc

import pandas as pd
import numpy as np
import networkx as nx
from pyarrow import csv

import ray

G = None
x = None
x_next = None
alpha = 0.15
num_workers = 3
actor_num = 2
graph_num = 9

@ray.remote(scheduling_strategy="SPREAD",num_cpus=1)
class workers:
        def __init__(self,G,alpha,x):
                self.graph = G
                self.alpha = alpha
                self.x = x

        def compute_x(self,n):
                tmp = self.alpha * sum([self.x[m]/self.graph.degree(m) for m in self.graph.neighbors(n)]) + (1-self.alpha)
                return (tmp,abs(tmp - self.x[n]),n)

ray.init()

# Get start time
start=time.time()

@ray.remote(scheduling_strategy="SPREAD",num_cpus=2)
def PageRank(i,alpha):

    parse_options = csv.ParseOptions(delimiter=" ")

    #edges = [ray.get(ref) for ref in ray.data.read_csv("hdfs://master:9000/user/hdoop/custom_graph/email-Eu-core.txt").to_numpy_refs()]
    edges = ray.data.read_csv(f"hdfs://master:9000/user/hdoop/Graph/edges/graph_edges_{i+1}.txt",parse_options=csv.ParseOptions(delimiter=" ")).to_pandas().drop_duplicates()
    edges.rename(columns={edges.columns[0]:"src",edges.columns[1]:"dst"},inplace=True)
    
    print("Building Graph...")

    G = nx.from_pandas_edgelist(edges,'src','dst')

    edges = None
    gc.collect()   

    G_ref = ray.put(G)

    print("Graph built!")

    x = (1-alpha) * np.ones(G.number_of_nodes())
    x_next = (1-alpha) * np.ones(G.number_of_nodes())

    nodes = G.nodes()

    print("Executing PageRank...")

    while True:

        x_ref = ray.put(x)
        worker = [workers.remote(G_ref,alpha,x_ref) for _ in range(actor_num)]

        compute_deg = ray.get([worker[i%actor_num].compute_x.remote(n) for i,n in enumerate(nodes)])

        err_arr = max([compute_deg[i][1] for i in range(len(compute_deg))])
        if err_arr < 1.0e-2:
            break
        print((err_arr))
        for i in range(len(compute_deg)):
            x_next[compute_deg[i][2]] = compute_deg[i][0]
        x = x_next.copy()
        for i in range(len(worker)):
            ray.kill(worker[i])


    print("Error:",err_arr)

    res = x
    res = list(enumerate(list(res)))

    total = sum([res[i][1] for i in range(len(res))])
    print(res[121][1]*len(res)/total) 

    res = sorted(res,key=lambda x: x[1],reverse=True)

    nodes = [res[i][0] for i in range(len(res))]
    score = [res[i][1] for i in range(len(res))]

    final = pd.DataFrame({'node':nodes,'score':score})

    final.head(10).to_csv("./results.csv",mode="a")
    print("DONE!")

for j in range(0,graph_num,num_workers):
    original_workers = [PageRank.remote(i,alpha) for i in range(j,j+num_workers)]
    ray.get(original_workers)

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
