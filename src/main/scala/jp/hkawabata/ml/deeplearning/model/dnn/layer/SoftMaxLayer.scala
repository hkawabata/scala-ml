package jp.hkawabata.ml.deeplearning.model.dnn.layer

import breeze.linalg.{*, Axis, DenseMatrix, sum}

class SoftMaxLayer extends Layer {
  var z: Option[DenseMatrix[Double]] = None

  def forward(in: DenseMatrix[Double]): DenseMatrix[Double] = {
    val tmp_a = in.map(x => math.exp(x))
    val tmp_v = sum(tmp_a, Axis._0)
    val out = (tmp_a.t(::, *) / tmp_v.inner).t
    z = Some(out.copy)
    out
  }

  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    val din = z.get *:* (dout.t(::, *) - sum(dout *:* z.get, Axis._0).inner).t
    z = None
    din
  }
}
