package jp.hkawabata.ml.deeplearning.model.dnn.layer

import breeze.linalg._

class SigmoidActivationLayer extends Layer {
  var sigmoid: Option[DenseMatrix[Double]] = None

  def forward(in: DenseMatrix[Double]): DenseMatrix[Double] = {
    sigmoid = Some(in.map(x => 1.0 / (1.0 + math.exp(-x))))
    sigmoid.get
  }

  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    println(dout.rows)
    println(dout.cols)

    val dphi: DenseMatrix[Double] = sigmoid.get.map(x => x * (1.0 - x))

    val din: DenseMatrix[Double] = dout *:* dphi

    sigmoid = None

    din
  }
}
