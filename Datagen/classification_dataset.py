import os
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification

from fastparquet import write

target_train = 20
target_test = 5

# We need data to fit into ram each time so we'll split it into chunks
chunk_size = 4*10**5

# we consider 60 features per sample, with informative being 20 of them
num_feats = 60
num_info = 20

# Create training data
for index in range(target_train):
	X,y = make_classification(n_samples=chunk_size, n_features=num_feats,  n_informative=num_info, random_state=42) 

	df_X = pd.DataFrame(X, columns=[f'feat_{i+1}' for i in range(num_feats)])
	df_y = pd.DataFrame(y, columns=['label'])
	data = pd.concat([df_X,df_y],axis=1)

	# for every 10**5 samples * num_feats features each, make 100 features null at random
	import random
	for i in range(100): 
		sample = random.randint(0,chunk_size-1)
		column = random.randint(1,num_feats)
		data[f'feat_{column}'][sample] = np.NaN

	write(f"./classification_data_train_{index}.parquet",data)

# Create testing data
for index in range(target_test):
        X,y = make_classification(n_samples=chunk_size, n_features=num_feats,  n_informative=num_info, random_state=42) 

        df_X = pd.DataFrame(X, columns=[f'feat_{i+1}' for i in range(num_feats)])
        df_y = pd.DataFrame(y, columns=['label'])
        data = pd.concat([df_X,df_y],axis=1)

        # for every 10**5 samples * num_feats features each, make 100 features null at random
        import random
        for i in range(100): 
                sample = random.randint(0,chunk_size-1)
                column = random.randint(1,num_feats)
                data[f'feat_{column}'][sample] = np.NaN

        write(f"./classification_data_test_{index}.parquet",data) 
