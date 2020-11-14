package jp.hkawabata.ml

import breeze.linalg.{DenseMatrix, DenseVector}
import jp.hkawabata.ml.deeplearning.model.dnn.layer.{AffineLayer, Layer, SigmoidActivationLayer}
import jp.hkawabata.ml.util.DataGenerator
import org.scalatest.wordspec.AnyWordSpec

class MyTest extends AnyWordSpec {
  "breeze test" in {
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
    val z = DenseMatrix.zeros[Double](3, 2)
    println("z\n" + z)
    println("a * b\n" + (a * b))
    println("b * a\n" + (b * a))
    println("a + b.t\n" + (a + b.t))
    println("a * 2.0\n" + (a * 2.0))
    println("a + 1.0\n" + (a + 1.0))
    println("a * v" + (a * v))

    println(DataGenerator.circle(4.0, 10))
    println(DataGenerator.circle(1.0, 10))

    println(a.reshape(3,2))

    println("a *:* b.t\n" + (a *:* b.t))

    println("---------")
    println(a.flatten())
    println(a.flatten().toDenseMatrix.reshape(2, 3))
    println("---------")

    // 行列を上下に結合
    println(DenseMatrix.vertcat(a, b.t))
    // 行列を左右に結合
    println(DenseMatrix.horzcat(a, b.t))
  }

  "DNN" in {
    val layers: Seq[Layer] = List(
      new AffineLayer(2, 4),
      new SigmoidActivationLayer,
      new AffineLayer(4, 3),
      new SigmoidActivationLayer
    )

    val data = DenseMatrix.horzcat(
      DataGenerator.circle(1, 10),
      DataGenerator.circle(2, 10, (1, 3)),
      DataGenerator.circle(3, 10, (3, -4))
    )
    val labels: DenseVector[Int] = DenseMatrix.horzcat(
      DenseMatrix.fill(1, 10)(1),
      DenseMatrix.fill(1, 10)(2),
      DenseMatrix.fill(1, 10)(3)
    ).toDenseVector

    var x: DenseMatrix[Double] = data.copy
    val y: DenseVector[Int] = labels.copy

    layers.foreach(l => x = l.forward(x))
    println(x)
    layers.reverseIterator.foreach(l => x = l.backward(x))
    println(x)
  }
}
