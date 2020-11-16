package jp.hkawabata.ml.deeplearning.model.dnn

import breeze.linalg.{Axis, DenseMatrix, argmax, sum}

import scala.collection.mutable.ListBuffer

class ModelEvaluator(val correctLabels: DenseMatrix[Double]) {
  val costBuff = new ListBuffer[Double]
  val precisionBuff = new ListBuffer[Double]

  private var errorOpt: Option[DenseMatrix[Double]] = None

  def evaluate(predictedLabels: DenseMatrix[Double]): Unit = {
    val cost = sum(
      - correctLabels *:* predictedLabels.map(x => math.log(x))
        - (1.0 - correctLabels) *:* predictedLabels.map(x => math.log(1.0 - x))
    )
    costBuff.append(cost)
    errorOpt = Some((predictedLabels - correctLabels) /:/ (predictedLabels *:* (1.0 - predictedLabels)) / correctLabels.cols.toDouble)

    val tmp = (argmax(predictedLabels, Axis._0) :== argmax(correctLabels, Axis._0)).inner.toDenseVector
    val precision = tmp.data.count(x => x).toDouble / tmp.size
    precisionBuff.append(precision)
  }

  def error: DenseMatrix[Double] = errorOpt.get
}
