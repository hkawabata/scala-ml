package jp.hkawabata.ml.deeplearning.model.dnn

import breeze.linalg.{Axis, DenseMatrix, argmax, sum}

import scala.collection.mutable.ListBuffer

class CostEvaluator(val correctLabels: DenseMatrix[Double]) {
  val costBuff = new ListBuffer[Double]
  val precisionBuff = new ListBuffer[Double]

  private var errorOpt: Option[DenseMatrix[Double]] = None

  def evaluate(predictedLabels: DenseMatrix[Double]): Unit = {
    val tmp = - correctLabels *:* predictedLabels.map(x => math.log(x))
    - (1.0 - correctLabels) *:* predictedLabels.map(x => math.log(1.0 - x))
    costBuff.append(sum(tmp))
    errorOpt = Some((predictedLabels - correctLabels) /:/ (predictedLabels *:* (1.0 - predictedLabels)) / correctLabels.cols.toDouble)
    val tmp2 = (argmax(predictedLabels, Axis._0) :== argmax(correctLabels, Axis._0)).inner.toDenseVector
    precisionBuff.append(tmp2.data.count(x => x).toDouble / tmp2.size)
  }

  def cost: Double = costBuff.last
  def error: DenseMatrix[Double] = errorOpt.get
}
