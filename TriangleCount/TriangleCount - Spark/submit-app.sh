spark-submit --class "TriangleCount" --master spark://master:7077 --jars graphframes-0.8.3-spark3.5-s_2.12.jar target/scala-2.12/trianglecount-with-spark_2.12-1.0.jar 3 8,8 8,8 8,8
