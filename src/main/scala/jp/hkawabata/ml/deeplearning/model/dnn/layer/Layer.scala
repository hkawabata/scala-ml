package jp.hkawabata.ml.deeplearning.model.dnn.layer

import breeze.linalg.DenseMatrix

abstract class Layer {
  def forward(in: DenseMatrix[Double]): DenseMatrix[Double]
  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double]
}
