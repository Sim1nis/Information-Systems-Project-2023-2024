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

class Execute_PageRank(i: Int, spark: SparkSession) extends Runnable {
  override def run(): Unit = {
    
    import spark.implicits._

    var vertices = spark.read.option("lineSep","\n").text("hdfs://master:9000/user/hdoop/Graph/vertices/graph_vertices.txt").withColumnRenamed("value","id")

    var edges = spark.read.option("lineSep", "\n").text("hdfs://master:9000/user/hdoop/Graph/edges/graph_edges_" + i.toString() + ".txt").dropDuplicates()
    var side_1 = edges.select(split(col("value")," ").getItem(0).as("src"),
    split(col("value")," ").getItem(1).as("dst"))
    .drop("value")
    
    var side_2 = side_1.select(col("dst").as("src"),col("src").as("dst"))
    
    edges = side_1.union(side_2)

    var graph = GraphFrame(vertices, edges)

    var results = graph.pageRank.resetProbability(0.85).tol(0.01).run()

    var resDF = results.vertices.select("id", "pagerank").orderBy(col("pagerank").desc)
    resDF = resDF.limit(10)

    resDF.write.mode(SaveMode.Append).csv("./results-spark.csv")
  }
}

object PageRank {

  def main(args: Array[String]): Unit = {

    //System.setErr(new PrintStream(new FileOutputStream("log.out")))
    //System.setOut(new PrintStream(new FileOutputStream("results.out")))

    val num_threads = (3).toInt;
    val num_graphs = (3).toInt;

    val before = System.currentTimeMillis;

    val spark = SparkSession.builder.appName("PageRank - Spark").getOrCreate()
    var th = new Array[Thread](num_threads);
    for (z<- 1 until num_graphs by num_threads) {
      for (i <- z until z+num_threads)  
        { 
            th(i%num_threads) = new Thread(new Execute_PageRank((i).toInt,spark)); 
            th(i%num_threads).setName(i.toString());

            th(i%num_threads).run(); 
        } 
      for (i <- z until z+num_threads)  
        { 
            th(i%num_threads).join()
        }
    }
        

    val totalTime=System.currentTimeMillis-before;    
 
    spark.stop()

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

