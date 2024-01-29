name := "TriangleCount with Spark"

version := "1.0"

scalaVersion := "2.12.18"

resolvers += "SparkPackages" at "https://repos.spark-packages.org/"

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % "3.5.0",
    "org.apache.spark" %% "spark-sql" % "3.5.0",
    "org.apache.spark" %% "spark-mllib-local" % "3.5.0",
    "org.apache.spark" % "spark-graphx_2.12" % "3.5.0",
    "graphframes" % "graphframes" % "0.8.1-spark3.0-s_2.12"
  )
