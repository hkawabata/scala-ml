package jp.hkawabata.ml

import breeze.linalg._
import breeze.plot._
import jp.hkawabata.ml.deeplearning.model.dnn.model.DNN
import jp.hkawabata.ml.util.DataGenerator
import org.jfree.chart.axis.NumberTickUnit
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
    val d: DenseMatrix[((Int, Int, Int), (Int, Int, Int))] = DenseMatrix(
      (
        (
          (1,2,3), (4,5,6)
        ),
        (
          (7,8,9), (0,1,2)
        )
      ),
      (
        (
          (3,4,5), (6,7,8)
        ),
        (
          (9,0,1), (2,3,4)
        )
      )
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
    println("---------- 3x2x2x2 配列 D ----------")
    println(d)
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
      DataGenerator.circle(1, 100, (1.6, 0.9)),
      DataGenerator.circle(1, 100, (1.6, -0.9))
    )
    val labels: List[String] = List.fill(100)("A") ++ List.fill(100)("B") ++ List.fill(100)("C")

    // モデルを学習
    val model = new DNN(eta = 0.5, numHiddenLayers = 2, numHiddenNodes = 5, epochs = 1000)
    model.fit(data, labels)

    // 学習過程を可視化
    val cost = DenseVector(model.getCost: _*)
    val precision = DenseVector(model.getPrecision: _*)
    val epochs: DenseVector[Double] = DenseVector((0 until precision.length).map(_.toDouble): _*)
    val f = Figure()
    val p1 = f.subplot(1, 2, 0)
    p1.title = "Precision"
    p1.xlabel = "epochs"
    p1.ylim(0, 1.0)
    p1.yaxis.setTickUnit(new NumberTickUnit(0.1))
    p1 += plot(epochs, precision, '-')
    val p2 = f.subplot(1, 2, 1)
    p2.title = "Cost"
    p2.xlabel = "epochs"
    p2 += plot(epochs, cost)
    f.saveas("target/dnn.png")
  }
}
