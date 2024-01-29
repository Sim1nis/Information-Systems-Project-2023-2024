spark-submit --class "PageRank" --master spark://master:7077 --executor-memory 12g --jars graphframes-0.8.3-spark3.5-s_2.12.jar target/scala-2.12/pagerank-with-spark_2.12-1.0.jar 3 8,2 8,2 8,2
