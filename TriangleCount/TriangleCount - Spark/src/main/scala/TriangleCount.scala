/* PageRank.scala */
import org.apache.spark.sql.SparkSession
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl._ 
import org.apache.spark.graphx.lib._ 
import org.apache.spark.graphx.util._
import org.apache.spark.sql._
import org.graphframes._
import org.apache.spark.sql.functions._

import java.io._ 

object TriangleCount {
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder.appName("PageRank - Spark").getOrCreate()

    val before = System.currentTimeMillis;
  
    import spark.implicits._

    var vertices = spark.read.option("lineSep","\n").text("hdfs://master:9000/user/hdoop/Graph/vertices").withColumnRenamed("value","id")
 
    var edges = spark.read.option("lineSep", "\n").text("hdfs://master:9000/user/hdoop/Graph/edges").distinct()

    edges = edges.select(split(col("value")," ").getItem(0).as("src"),
    split(col("value") ," ").getItem(1).as("dst"))
    .drop("value")
    
    var graph = GraphFrame(vertices, edges)

    var result = graph.triangleCount.run()
    val sumRes = result.agg(sum("count")).show()
    
    spark.stop()
    
    val totalTime=System.currentTimeMillis-before;    
 
    val num_workers = args(0);
    
    val fw = new FileWriter("spark_stats.txt", true); 
    
    fw.write("Total time: " + (totalTime/1000).toString + "\n");
    fw.write("Total workers: " + num_workers.toString + "\n");

    for( i <- 1 to args.length-1) {
        var stats = args(i).split(",");
        fw.write("   Worker: " + i.toString + "\n");    
	fw.write("      Memory: " + stats(0) + "G\n");
        fw.write("      CPU Cores: " + stats(1) + "\n"); 
    }
    fw.write("\n");
    fw.close();   
  }
}
