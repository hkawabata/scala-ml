package jp.hkawabata.ml.util

import breeze.linalg.DenseMatrix

class LabelManager[T](val labels: List[T]) {
  private val uniqueLabels: List[T] = labels.distinct
  private val label2IndexMap: Map[T, Int] = uniqueLabels.zipWithIndex.toMap
  val labelMatrix: DenseMatrix[Double] = {
    val result = DenseMatrix.zeros[Double](uniqueLabels.size, labels.size)
    labels.zipWithIndex.foreach{
      case (label, i) =>
        result(label2Index(label), i) = 1.0
    }
    result
  }

  def label2Index(label: T): Int = label2IndexMap(label)

  def index2Label(index: Int): T = uniqueLabels(index)

  def numUniqueLabels: Int = uniqueLabels.size
}
