from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import os

from fastparquet import write

target = 11*1024**10*20

# We need data to fit into ram each time so we'll split it into chunks
chunk_size = 5*10**5

# we consider 40 features per sample and 5 centers, cluster_std = 0.3
num_feats = 40
num_centers = 5
clust_std = 0.3

# Create training data
current_size = 0
index = 0

while current_size < target:
	features, clusters = make_blobs(n_samples = chunk_size, n_features = num_feats, centers = num_centers, cluster_std = clust_std, shuffle = True)

	df_X = pd.DataFrame(features, columns=[f'feat_{i+1}' for i in range(num_feats)])
	df_y = pd.DataFrame(clusters,columns=["cluster"])
	data = pd.concat([df_X,df_y],axis=1)
	
	write(f"./clustering_data_{index}.parquet",data)
		
	current_size += os.stat(f"./clustering_data_{index}.parquet").st_size 

	index += 1
	
	print("Done 1 iter")
