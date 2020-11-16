package jp.hkawabata.ml.deeplearning.model.dnn.layer

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, sum}
import breeze.stats.distributions.Gaussian

class AffineLayer(nFeatures: Int, nNewFeatures: Int) extends Layer {
  private val W: DenseMatrix[Double] = DenseMatrix.rand(nNewFeatures, nFeatures, Gaussian(0, 1))
  private val b: DenseVector[Double] = DenseVector.rand(nNewFeatures, Gaussian(0, 1))
  private var x: Option[DenseMatrix[Double]] = None

  def forward(in: DenseMatrix[Double], isTraining: Boolean): DenseMatrix[Double] = {
    if (isTraining) x = Some(in.copy)
    val tmp = W * in
    tmp(::, *) + b
  }

  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    val din: DenseMatrix[Double] = W.t * dout
    val dW: DenseMatrix[Double] = dout * x.get.t
    val db: DenseVector[Double] = sum(dout, Axis._1)

    W -= dW
    b -= db

    x = None

    din
  }
}
