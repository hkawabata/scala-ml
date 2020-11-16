name := "scala-ml"

version := "0.1"

scalaVersion := "2.12.7"

libraryDependencies ++= Seq(
  // https://mvnrepository.com/artifact/org.scalanlp/breeze
  "org.scalanlp" %% "breeze" % "1.0",
  // https://mvnrepository.com/artifact/org.scalanlp/breeze-viz
  "org.scalanlp" %% "breeze-viz" % "1.0",
  // https://mvnrepository.com/artifact/org.scalatest/scalatest
  "org.scalatest" %% "scalatest" % "3.2.3" % Test
)