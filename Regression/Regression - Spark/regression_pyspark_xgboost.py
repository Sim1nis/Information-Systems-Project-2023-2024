from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from xgboost.spark import SparkXGBRegressor
#from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorIndexer, VectorAssembler, Imputer
from pyspark.ml.evaluation import RegressionEvaluator

import os
import sys
import time

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ["SPARK_HOME"] = "/opt/spark/"

spark = SparkSession \
     .builder \
     .master("spark://master:7077").getOrCreate()

print("spark session created")

# Get start time
start = time.time()

# Read dataframes
trainingData = spark.read.option("header", "true").option("inferSchema", "true").parquet("hdfs://master:9000/user/hdoop/Regression/train/")
testData = spark.read.option("header", "true").option("inferSchema", "true").parquet("hdfs://master:9000/user/hdoop/Regression/test/")

# we will use all columns except for one (and of course not the index)
num_features = len(trainingData.columns) - 2

imputer = Imputer(
    inputCols = [f"feat_{i+1}" for i in range(num_features)],
    outputCols = [f"feat_{i+1}_imputed" for i in range(num_features)]
).setStrategy("mean")

# Assemble columns to one
assembler =\
    VectorAssembler(inputCols=[f"feat_{i+1}_imputed" for i in range(num_features)], outputCol="assembledFeatures")


# Create Linear (Ridge) Regressor
#lr = LinearRegression(featuresCol = 'assembledFeatures', labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0,solver='l-bfgs',tol=0.0001)

sre = SparkXGBRegressor(
  n_estimators=100,
  learning_rate=0.01,
  features_col="assembledFeatures",
  label_col="label",
  max_depth=6,
  num_workers=3
)

# Declare the pipeline
pipeline = Pipeline(stages=[imputer,assembler,sre])

# Create model and fit data
start_training = time.time()
model = pipeline.fit(trainingData)
stop_training = time.time()

# Make predictions
predictions = model.transform(testData)

# Evaluate model
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# Get end time
end = time.time()

# Write stats about time, workers, memory and cores to file
num_workers = sys.argv[1]
worker_info = []
for i in range(2,len(sys.argv)):
	worker_info.append(sys.argv[i].split(","))

out = open("spark_stats.txt","a")

out.write(f"Root Mean Squared Error (RMSE) on test data = {rmse}\n")
out.write(f"Total Time: {end-start}\n")
out.write(f"Training Time: {stop_training-start_training}\n")
out.write(f"Total Workers: {num_workers}\n")

for i,worker in enumerate(worker_info):
	out.write(f"   Worker {i+1}:\n")
	out.write(f"      Memory: {worker[0]}G\n")
	out.write(f"      CPU Cores: {worker[1]}\n")

out.write("\n")
out.close()
