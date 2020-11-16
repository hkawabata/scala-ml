package jp.hkawabata.ml.deeplearning.model.dnn.layer

import breeze.linalg.{*, Axis, DenseMatrix, sum}

class SoftMaxLayer extends Layer {
  private var z: Option[DenseMatrix[Double]] = None

  def forward(in: DenseMatrix[Double], isTraining: Boolean): DenseMatrix[Double] = {
    val tmp_a = in.map(x => math.exp(x))
    val tmp_v = sum(tmp_a, Axis._0)
    val out = tmp_a(*, ::) / tmp_v.inner
    if (isTraining) z = Some(out)
    out
  }

  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    val din = z.get *:* (dout(*, ::) - sum(dout *:* z.get, Axis._0).inner)
    z = None
    din
  }
}
