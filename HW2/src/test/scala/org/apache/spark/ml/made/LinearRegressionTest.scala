package org.apache.spark.ml.made

import org.apache.spark.ml.linalg.{Vector, Vectors}
import breeze.linalg._
import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should
import org.apache.spark.ml.{Pipeline, PipelineModel}
import com.google.common.io.Files
import org.scalatest._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta: Double = 1
  lazy val dataset_features: DataFrame = LinearRegressionTest._dataset_features
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val X_train: DenseMatrix[Double] = LinearRegressionTest.X_train
  lazy val coefficient: DenseVector[Double] = LinearRegressionTest.coefficient
  lazy val y_train: DenseVector[Double] = LinearRegressionTest.y_train
  val bias: Double = 0.0

  "Model" should "predict by input data" in {
    val model: LinearRegressionModel = new LinearRegressionModel(Vectors.fromBreeze(coefficient), bias)
      .setInputCol("features")
      .setOutputCol("predict")

    val predict: DataFrame = model.transform(dataset_features)

    predict.columns.length should be(2)

    predict.select("predict").collectAsList().get(0)(0) should be(y_train(0))
  }

  "Estimator" should "calculate weights" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("predict")

    val model = estimator.fit(data)

    println(model.coefficients)
    model.coefficients(0) should be(coefficient(0) +- delta)
    model.coefficients(1) should be(coefficient(1) +- delta)
    model.coefficients(2) should be(coefficient(2) +- delta)

    validateModel(model, model.transform(dataset_features))
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("predict")
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    validateModel(model, model.transform(dataset_features))
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("predict")
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(dataset_features))
  }

  private def validateModel(model: LinearRegressionModel, data: DataFrame) = {
    val predict = data.select("predict").collectAsList()

    predict.get(0)(0) should equal(y_train(0) +- delta)
    predict.get(1)(0) should equal(y_train(1) +- delta)
  }

}

object LinearRegressionTest extends WithSpark {
  lazy val X_train: DenseMatrix[Double] = DenseMatrix.rand(100000, 3)
  lazy val coefficient: DenseVector[Double] = DenseVector(1.5, 0.3, -0.7)
  lazy val y_train: DenseVector[Double] = X_train * coefficient

  import spark.implicits._

  lazy val _dataset_features: DataFrame = X_train(*, ::)
    .iterator
    .map(x => Tuple1(Vectors.fromBreeze(x)))
    .toSeq
    .toDF("features")

  lazy val _dataset: DenseMatrix[Double] = DenseMatrix.horzcat(X_train, y_train.asDenseMatrix.t)
  lazy val _data: DataFrame = _dataset(*, ::)
    .iterator
    .map(x => Tuple1(Vectors.fromBreeze(x)))
    .toSeq
    .toDF("features")
}
