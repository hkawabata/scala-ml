package jp.hkawabata.ml

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, det, eig, inv, sum}
import jp.hkawabata.ml.deeplearning.model.dnn.CostEvaluator
import jp.hkawabata.ml.deeplearning.model.dnn.layer.{AffineLayer, Layer, SigmoidActivationLayer, SoftMaxLayer}
import jp.hkawabata.ml.util.{DataGenerator, LabelManager}
import org.scalatest.wordspec.AnyWordSpec

class MyTest extends AnyWordSpec {
  "breeze 行列演算" in {
    println("\n==================== breeze 行列演算 ====================\n")

    val v = DenseVector(1.0, 2.0, 3.0)

    val a = DenseMatrix(
      (1.0, 2.0, 3.0),
      (2.0, 5.0, 1.0)
    )
    val b = DenseMatrix(
      (2.0, 3.0),
      (4.0, 7.0),
      (-3.0, 2.0)
    )
    val c = DenseMatrix(
      (1.0, 2.0, 3.0),
      (1.0, 4.0, 2.0),
      (4.0, 3.0, 5.0)
    )
    val i = DenseMatrix.eye[Double](3)
    val one = DenseMatrix.ones[Double](2, 3)
    val z = DenseMatrix.zeros[Double](4, 5)

    println("---------- 2x3 行列 A ----------")
    println(a)
    println("---------- 3x2 行列 B ----------")
    println(b)
    println("---------- 3x3 行列 C ----------")
    println(c)
    println("---------- ベクトル v ----------")
    println(v)
    println("---------- 単位行列 I ----------")
    println(i)
    println("---------- 全てが1の行列 ----------")
    println(one)
    println("---------- ゼロ行列 Z ----------")
    println(z)
    println("---------- ゼロ行列 Z の値を更新 ----------")
    z(0, 1) = 1.0
    z(::, 0) := 2.0
    z(1 to 2, 2 to 3) := 3.0
    println(z)
    println("---------- 行列の和 ----------")
    println(a + b.t)
    println("---------- 行列の積 ----------")
    println(a * b)
    println(b * a)
    println("---------- 行列の要素同士の積 ----------")
    println(a *:* b.t)
    println("---------- 要素の定数倍 ----------")
    println(a * 2.0)
    println("---------- 要素に定数を足す ----------")
    println(a + 1.0)
    println("---------- 行列の全ての列に同じベクトルを足す ----------")
    println(s"$b, $v\n-->\n${b(::, *) + v}")
    println("---------- 行列の全ての行に同じベクトルを足す ----------")
    println(s"$a, $v\n-->\n${a(*, ::) + v}")
    println("---------- 行列をベクトルにかける ----------")
    println(a * v)
    println("---------- C の逆行列 ----------")
    println(s"$c\nx\n${inv(c)}\n=\n${c * inv(c)}")
    println("---------- C の行列式 ----------")
    println(det(c))
    println("---------- C の固有値、固有ベクトル ----------")
    println(eig(c).eigenvalues)
    println(eig(c).eigenvectors)

    println("---------- 要素全ての和を取る ----------")
    println(s"$a\n--> ${sum(a)}")
    println("---------- 列で和を取る ----------")
    println(s"$a\n--> ${sum(a, Axis._0)}")
    println("---------- 行で和を取る ----------")
    println(s"$a\n--> ${sum(a, Axis._1)}")

    println("---------- 行列を上下に結合 ----------")
    println(DenseMatrix.vertcat(a, b.t))
    println("---------- 行列を左右に結合 ----------")
    println(DenseMatrix.horzcat(a, b.t))

    println("---------- 行列の整形 ----------")
    println(a.reshape(3,2))

    println("---------- 行列をフラットに展開してベクトルに → reshape で元に戻す ----------")
    println(a.flatten())
    println(a.flatten().toDenseMatrix.reshape(2, 3))
  }

  "DataGenerator" in {
    println("\n==================== DataGenerator ====================\n")

    println("---------- 中心 (0, 0), 半径4の円 ----------")
    println(DataGenerator.circle(4.0, 10))
    println("---------- 中心 (10, 20), 半径1の円 ----------")
    println(DataGenerator.circle(1.0, 10, (10.0, 20.0)))
    println("---------- 左下 (0, 0), 幅10, 高さ5の長方形 ----------")
    println(DataGenerator.rectangle(10, 5, 10))
    println("---------- 左下 (10, 20), 幅10, 高さ5の長方形 ----------")
    println(DataGenerator.rectangle(10, 5, 10, (10.0, 20.0)))
  }

  "DNN" in {
    println("\n==================== DNN ====================\n")

    // データ作成
    val data = DenseMatrix.horzcat(
      DataGenerator.circle(1, 100),
      DataGenerator.circle(1, 100, (1.5, 0.9)),
      DataGenerator.circle(1, 100, (1.5, -0.9))
    )
    val labels: List[String] = List.fill(100)("A") ++ List.fill(100)("B") ++ List.fill(100)("C")

    // 前処理
    val labelManager = new LabelManager[String](labels)
    val y: DenseMatrix[Double] = labelManager.labelMatrix

    // モデルアーキテクチャの定義
    val eta: Double = 1.0
    val numHiddenNodes: Int = 5
    val layers: Seq[Layer] = List(
      new AffineLayer(2, numHiddenNodes),
      new SigmoidActivationLayer,
      new AffineLayer(numHiddenNodes, labelManager.numUniqueLabels),
      new SigmoidActivationLayer,
      new SoftMaxLayer
    )
    val costEvaluator = new CostEvaluator(y)

    // 学習
    (0 to 200).foreach {
      t =>
        var out = data.copy
        layers.foreach(l => out = l.forward(out))
        costEvaluator.evaluate(out)
        var error = costEvaluator.error * eta
        if (t % 10 == 0) {
          println(s"cost : ${costEvaluator.costBuff.last}, precision: ${costEvaluator.precisionBuff.last}")
        }
        layers.reverseIterator.foreach(l => error = l.backward(error))
    }

    // precision の推移を可視化
    val size = costEvaluator.precisionBuff.size
    costEvaluator.precisionBuff.result.zipWithIndex.foreach{
      case (p, i) =>
        if (i % (size / 20) == 0) {
          val n = (p * 50).toInt
          println("*" * n + "." * (50 - n))
        }
    }
  }
}
