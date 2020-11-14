package jp.hkawabata.ml.deeplearning.model.dnn.layer

import breeze.linalg.DenseMatrix

class SoftMaxLayer extends Layer {
  def forward(in: DenseMatrix[Double]): DenseMatrix[Double] = ???

  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = ???
}
