package com.hongya.bigdata.day06

/**
  * Created by allanhwang on 17-5-16.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd._
import org.jfree.data.category.DefaultCategoryDataset
import org.joda.time.{Duration, _}

object AlsEvaluation {
  val data_path = "/Users/dengziming/Documents/hongya/data/day06/data.txt"
  val item_path = "/Users/dengziming/Documents/hongya/data/day06/item.txt"


  def main(args: Array[String]) {
    SetLogger
    println("==========数据准备阶段===============")
    val (trainData, validationData, testData, movieTitle) = PrepareData()
    trainData.persist(); validationData.persist(); testData.persist()
    println("==========训练验证阶段===============")
    val bestModel = trainValidation(trainData, validationData)
    println("==========测试阶段===============")
    val testRmse = computeRMSE(bestModel, testData)
    println("使用testData测试bestModel," + "结果rmse = " + testRmse)
    trainData.unpersist(); validationData.unpersist(); testData.unpersist()
  }

  def trainValidation(trainData: RDD[Rating], validationData: RDD[Rating]): MatrixFactorizationModel = {
    println("-----评估 rank参数使用 ---------")
    evaluateParameter(trainData, validationData, "rank", Array(5, 10, 15, 20, 50, 100), Array(10), Array(0.1))
    println("-----评估 numIterations ---------")
    evaluateParameter(trainData, validationData, "numIterations", Array(10), Array(5, 10, 15, 20, 25), Array(0.1))
    println("-----评估 lambda ---------")
    evaluateParameter(trainData, validationData, "lambda", Array(10), Array(10), Array(0.05, 0.1, 1, 5, 10.0))
    println("-----所有参数交叉评估找出最好的参数组合---------")
    val bestModel = evaluateAllParameter(trainData, validationData, Array(5, 10, 15, 20, 25), Array(5, 10, 15, 20, 25), Array(0.05, 0.1, 1, 5, 10.0))
    bestModel
  }

  def evaluateParameter(trainData: RDD[Rating], validationData: RDD[Rating], evaluateParameter: String,
                        rankArray: Array[Int], numIterationsArray: Array[Int], lambdaArray: Array[Double]) = {
    val dataBarChart = new DefaultCategoryDataset()

    val dataLineChart = new DefaultCategoryDataset()
    for (rank <- rankArray; numIterations <- numIterationsArray; lambda <- lambdaArray) {

      val (rmse, time) = trainModel(trainData, validationData, rank, numIterations, lambda)

      val parameterData =
        evaluateParameter match {
          case "rank"          => rank;
          case "numIterations" => numIterations;
          case "lambda"        => lambda
        }
      dataBarChart.addValue(rmse, evaluateParameter, parameterData.toString())
      dataLineChart.addValue(time, "Time", parameterData.toString())
    }

    Chart.plotBarLineChart("ALS evaluations " + evaluateParameter, evaluateParameter, "RMSE", 0.58, 5, "Time", dataBarChart, dataLineChart)
  }

  def evaluateAllParameter(trainData: RDD[Rating], validationData: RDD[Rating],
                           rankArray: Array[Int], numIterationsArray: Array[Int], lambdaArray: Array[Double]): MatrixFactorizationModel = {
    val evaluations =
      for (rank <- rankArray; numIterations <- numIterationsArray; lambda <- lambdaArray) yield {
        val (rmse, time) = trainModel(trainData, validationData, rank, numIterations, lambda)
        (rank, numIterations, lambda, rmse)
      }
    val Eval = evaluations.sortBy(_._4)
    val BestEval = Eval(0)
    println("最佳model参数：rank:" + BestEval._1 + ",iterations:" + BestEval._2 + "lambda" + BestEval._3 + ",结果rmse = " + BestEval._4)
    ALS.train(trainData, BestEval._1, BestEval._2, BestEval._3)
  }
  def PrepareData(): (RDD[Rating], RDD[Rating], RDD[Rating],Map[Int,String]) = {

    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[4]"))
    //----------------------1.创建用户评分数据-------------
    print("开始读取用户评分数据...")
    val rawUserData = sc.textFile(data_path)

    val rawRatings = rawUserData.map(_.split("\t").take(3))

    val ratingsRDD = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    println(s"共计：${ratingsRDD.count}条ratings")

    //----------------------2.创建电影ID与名称对照表-------------
    print("开始读取电影数据...")
    val itemRDD = sc.textFile(item_path)
    val movieTitle = itemRDD.map(line => line.split("\\|").take(2))
      .map(array => (array(0).toInt, array(1))).collect().toMap
    //----------------------3.显示数据记录数-------------
    val numRatings = ratingsRDD.count()
    val numUsers = ratingsRDD.map(_.user).distinct().count()
    val numMovies = ratingsRDD.map(_.product).distinct().count()
    println("共计：ratings: " + numRatings + " User " + numUsers + " Movie " + numMovies)
    //----------------------4.以随机方式将数据分为3个部分并且返回-------------
    println("将数据分为")
    val Array(trainData, validationData, testData) = ratingsRDD.randomSplit(Array(0.8, 0.2, 0.1))

    println("  trainData:" + trainData.count() + "  validationData:" + validationData.count() + "  testData:" + testData.count())
    (trainData, validationData, testData,movieTitle)
  }

  def trainModel(trainData: RDD[Rating], validationData: RDD[Rating], rank: Int, iterations: Int, lambda: Double): (Double, Double) = {
    val startTime = new DateTime()
    val model = ALS.train(trainData, rank, iterations, lambda)
    val endTime = new DateTime()
    val Rmse = computeRMSE(model, validationData)
    val duration = new Duration(startTime, endTime)
    println(f"训练参数：rank:$rank%3d,iterations:$iterations%.2f ,lambda = $lambda%.2f 结果 Rmse=$Rmse%.2f" + "训练需要时间" + duration.getMillis + "毫秒")
    (Rmse, duration.getStandardSeconds)
  }

  def computeRMSE(model: MatrixFactorizationModel, RatingRDD: RDD[Rating]): Double = {

    val num = RatingRDD.count()
    val predictedRDD = model.predict(RatingRDD.map(r => (r.user, r.product)))
    val predictedAndRatings =
      predictedRDD.map(p => ((p.user, p.product), p.rating))
        .join(RatingRDD.map(r => ((r.user, r.product), r.rating)))
        .values
    math.sqrt(predictedAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / num)
  }

  def SetLogger = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }

}