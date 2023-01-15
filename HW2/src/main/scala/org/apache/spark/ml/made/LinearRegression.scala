package org.apache.spark.ml.made

import breeze.linalg.{sum, DenseVector, DenseMatrix}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}


trait LinearRegressorParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val useBias = new BooleanParam(
    this, "useBias", "Whenever wasn't bias coefficient in model")

  def isUseBias : Boolean = $(useBias)
  def setUseBias(value: Boolean) : this.type = set(useBias, value)

  setDefault(useBias -> true)

  // method to check scheme
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}


class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressorParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  //key method of the Estimator
  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    val lr: Double = 0.1
    val maxiter: Int = 100
    val eps: Double = 0.001  // for breakpoint to model not over fitting (until not use)

    // convert dataframe to DenseMatrix
    val featuresTrain = dataset.columns
    val rows = dataset.count().toInt

    val newFeatureArray: Array[Double] = featuresTrain
      .indices
      .flatMap(i => dataset
        .select(featuresTrain(i))
        .collect())
      .map(r => r.toSeq.toArray).toArray.flatten.flatMap(_.asInstanceOf[org.apache.spark.ml.linalg.DenseVector].values)

    val newFeatureBreeze: DenseVector[Double] = new DenseVector[Double](newFeatureArray)
    val y_train: DenseVector[Double] = newFeatureBreeze(3 to newFeatureBreeze.size - 1 by 4)

    val x1: DenseVector[Double] = newFeatureBreeze(0 to newFeatureBreeze.size - 4 by 4)
    val x2: DenseVector[Double] = newFeatureBreeze(1 to newFeatureBreeze.size - 3 by 4)
    val x3: DenseVector[Double] = newFeatureBreeze(2 to newFeatureBreeze.size - 2 by 4)
    val X_train: DenseMatrix[Double] = DenseMatrix.horzcat(x1.asDenseMatrix.t, x2.asDenseMatrix.t, x3.asDenseMatrix.t)

    val countFeatures: Int = X_train.cols

    var weight: DenseVector[Double] = DenseVector.rand(countFeatures)
    var bias: Double = 0.0

    for (_ <- 1 to maxiter) {
      val loss = X_train * weight + bias - y_train
      val grad = X_train.t * loss
      weight -= lr * grad /:/ (1.0 * rows)
      bias -= lr * sum(loss) / rows
    }

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(weight), bias))
  }

  def copy(extra: ParamMap) = defaultCopy(extra)

  def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                          val uid: String,
                                          val coefficients: Vector,
                                          val intercept: Double) extends Model[LinearRegressionModel] with LinearRegressorParams with MLWritable {

  def this(coefficients: Vector, intercept: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), coefficients, intercept)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(coefficients, intercept), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bCoef = coefficients.asBreeze
    val transformUdf = if (isUseBias) {
      dataset.sqlContext.udf.register(uid + "_transform",
        (x: Vector) => {
          x.asBreeze.dot(bCoef) + intercept
        })
    } else {
      dataset.sqlContext.udf.register(uid + "_transform",
        (x: Vector) => {
          x.asBreeze.dot(bCoef)
        })
    }

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val params = coefficients -> intercept

      sqlContext.createDataFrame(Seq(params)).write.parquet(path + "/weigth")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val params = sqlContext.read.parquet(path + "/weigth")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val weight = params.select(params("_1").as[Vector]).first()
      val bias = params.select(params("_2")).first().getDouble(0)

      val model = new LinearRegressionModel(weight, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}

