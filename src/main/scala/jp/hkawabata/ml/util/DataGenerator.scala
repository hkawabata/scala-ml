package jp.hkawabata.ml.util

import breeze.linalg.DenseMatrix
import scala.util.Random

object DataGenerator {
  def circle(r: Double, n: Int, center: (Double, Double) = (0, 0)): DenseMatrix[Double] = {
    val radius: Seq[Double] = (1 to n).map(_ => Random.nextDouble() * r)
    val theta: Seq[Double] = (1 to n).map(_ => Random.nextDouble() * math.Pi * 2.0)
    val xy = radius.zip(theta).map{case (r_, t_) => (r_ * math.cos(t_) + center._1, r_ * math.sin(t_) + center._2)}
    DenseMatrix(xy:_*).t
  }

  def rectangle(a: Double, b: Double, n: Int, bottomLeft: (Double, Double) = (0, 0)): DenseMatrix[Double] = {
    val xy = (1 to n).map(_ => (Random.nextDouble() * a + bottomLeft._1, Random.nextDouble() * b + bottomLeft._2))
    DenseMatrix(xy:_*).t
  }
}
