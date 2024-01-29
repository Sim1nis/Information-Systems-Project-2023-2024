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

object PageRank {
  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession.builder.appName("PageRank - Spark").getOrCreate()
    
    val before = System.currentTimeMillis;

    import spark.implicits._

    var vertices = spark.read.option("lineSep","\n").text("hdfs://master:9000/user/hdoop/Graph/vertices").withColumnRenamed("value","id")

    var edges = spark.read.option("lineSep", "\n").text("hdfs://master:9000/user/hdoop/Graph/edges")

    var side_1 = edges.select(split(col("value")," ").getItem(0).as("src"),
    split(col("value")," ").getItem(1).as("dst"))
    .drop("value")
    
    var side_2 = side_1.select(col("dst").as("src"),col("src").as("dst"))
    
    edges = side_1.union(side_2)

    var graph = GraphFrame(vertices, edges)

    val results = graph.pageRank.resetProbability(0.85).maxIter(10).run()

/*    var gx: Graph[Row, Row] = graph.toGraphX
    gx.pageRank(0.0001).vertices
	.sortBy(-_._2).toDF
	.withColumnRenamed("_1","VertexId")
	.withColumnRenamed("_2","PageRank")
        .createOrReplaceTempView("pagerank")
*/

    results.vertices.select("id", "pagerank").orderBy(col("pagerank").desc).show()

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

