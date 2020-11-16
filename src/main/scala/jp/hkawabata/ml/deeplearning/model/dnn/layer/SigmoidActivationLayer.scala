package jp.hkawabata.ml.deeplearning.model.dnn.layer

import breeze.linalg.DenseMatrix

class SigmoidActivationLayer extends Layer {
  private var sigmoid: Option[DenseMatrix[Double]] = None

  def forward(in: DenseMatrix[Double], isTraining: Boolean): DenseMatrix[Double] = {
    val out = in.map(x => 1.0 / (1.0 + math.exp(-x)))
    if (isTraining) sigmoid = Some(out)
    out
  }

  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    val dphi: DenseMatrix[Double] = sigmoid.get.map(x => x * (1.0 - x))
    val din: DenseMatrix[Double] = dout *:* dphi
    sigmoid = None
    din
  }
}
