package jp.hkawabata.ml.deeplearning.model.dnn.model

import breeze.linalg.{Axis, DenseMatrix, DenseVector, argmax}
import jp.hkawabata.ml.deeplearning.model.dnn.CostEvaluator
import jp.hkawabata.ml.deeplearning.model.dnn.layer.{AffineLayer, Layer, SigmoidActivationLayer, SoftMaxLayer}
import jp.hkawabata.ml.util.LabelManager

import scala.collection.mutable.ListBuffer

/**
  * @param eta 学習率
  * @param numHiddenLayers 隠れ層の数
  * @param numHiddenNodes 隠れ層のノード数
  * @param epochs エポック数
  */
class DNN(val eta: Double, val numHiddenLayers: Int, val numHiddenNodes: Int, val epochs: Int) {

  private var cost: List[Double] = _
  private var precision: List[Double] = _
  private var layers: List[Layer] = _
  private var lm: LabelManager[String] = _

  def fit(data: DenseMatrix[Double], labels: List[String]): Unit = {
    val nFeatures = data.rows

    // ラベルの前処理
    lm = new LabelManager[String](labels)
    val y: DenseMatrix[Double] = lm.labelMatrix
    val nLabels = y.rows

    // 評価器
    val costEvaluator = new CostEvaluator(y)

    // 学習アーキテクチャの定義
    val layersBuff = new ListBuffer[Layer]
    layersBuff.append(
      new AffineLayer(nFeatures, numHiddenNodes),
      new SigmoidActivationLayer
    )
    for (i <- 0 until (numHiddenLayers - 1)) {
      layersBuff.append(
        new AffineLayer(numHiddenNodes, numHiddenNodes),
        new SigmoidActivationLayer
      )
    }
    layersBuff.append(
      new AffineLayer(numHiddenNodes, nLabels),
      new SigmoidActivationLayer,
      new SoftMaxLayer
    )
    layers = layersBuff.result()

    // 学習
    for (i <- 0 until epochs) {
      val out = forward(data, isTraining = true)
      costEvaluator.evaluate(out)
      backward(costEvaluator.error * eta)
    }

    cost = costEvaluator.costBuff.result()
    precision = costEvaluator.precisionBuff.result()
  }

  def predict(data: DenseMatrix[Double]): List[String] = {
    val out = forward(data, isTraining = false)
    argmax(out, Axis._0).inner.toArray.map(lm.index2Label).toList
  }

  def forward(data: DenseMatrix[Double], isTraining: Boolean): DenseMatrix[Double] = {
    var out = data.copy
    layers.foreach(l => out = l.forward(out, isTraining))
    out
  }

  def backward(error: DenseMatrix[Double]): Unit = {
    var err = error.copy
    layers.reverseIterator.foreach(l => err = l.backward(err))
  }

  def getCost: List[Double] = cost
  def getPrecision: List[Double] = precision
}
