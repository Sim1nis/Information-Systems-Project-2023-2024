This folder contains the scripts for each experiment.

Each experiment (Classification, Clustering, Regression, PageRank) is split into two folders: One for Ray and another one for Spark, containing the respective implementations of the experiment algorithms

For the execution of python scripts, the command syntax is as follows (unless specified elsewise):

```shell
python3 {experiment_name}_{ray|spark}.py {number_of_executors/nodes} [list of (memory_in_GB,number_of_cores) per executor]
e.g. python3 regression_ray.py 2 8,4 8,4
```

Output --> {ray|spark}_stats.txt
