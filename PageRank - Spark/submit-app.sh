spark-submit --class "PageRank" --executor-memory 12g  --master spark://master:7077 --jars graphframes-0.8.3-spark3.5-s_2.12.jar target/scala-2.12/pagerank-with-spark_2.12-1.0.jar 2 12,4 12,4
