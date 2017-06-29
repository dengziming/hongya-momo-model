package com.hongya.bigdata.day07

/**
  * Created by allanhwang on 17-5-9.
  */
import java.util.Properties

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd._
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

object Recommend {

  lazy val url = "jdbc:mysql://localhost:3306/test"
  lazy val username = "root"
  lazy val password = "123456"
  val data_path = "/Users/dengziming/Documents/hongya/data/day07/actionlist2.txt"

  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[1]"))

    // 开始读取用户评分数据...
    val rawUserData = sc.textFile(data_path)
    val rawRatings = rawUserData.map(_.split("\t").take(3))
    val ratingsRDD = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    println(s"共计：${ratingsRDD.count}条ratings")

    val model = ALS.train(ratingsRDD, 5, 20, 0.1)
    saveToMysql(model)
  }


  /**
    * 计算准确率，使用测试数据验证
    * 对于每一个验证数据的一条记录 user,products,rating，我们得到对这个用户的推荐结果，如果包含这条记录就算是推荐正确
    * */
  def accuracyRate(model: MatrixFactorizationModel, ratingRDD: RDD[Rating]): Double = {

    val total = ratingRDD.count().toDouble

    val users = ratingRDD.map(_.user).collect()
    val accuracy =  ratingRDD.map{rating =>

      (
        rating.product, //这条数据中的product
        model.recommendProducts(rating.user,10).map(_.product)//所有推荐的product
      )

    }.filter(tuple => tuple._2.contains(tuple._1)).count().toDouble

    accuracy / total

  }

  /**
    *
    * @param model model
    */
  def saveToMysql(model : MatrixFactorizationModel): Unit ={
    val productsForUsers: RDD[(Int, Array[Rating])] = model.recommendProductsForUsers(1)

    // 推荐结果
    val recommend: RDD[(Int, Int)] = productsForUsers.map{
      case (user:Int,rating:Array[Rating]) =>
        (user,rating.map(_.product))
    }.map{
      case (user:Int,pros:Array[Int]) =>
        (user,
          if (pros.length > 0) pros(0) else 999999)
    }

    val rowRDD = recommend.map{line =>
      Row(line._1,line._2)
    }
    val fields: Array[StructField] = Array(
      StructField("user",IntegerType),
      StructField("recommend",IntegerType))
    )
    val sc = recommend.sparkContext
    val sqlContext = new SQLContext(sc)
    val rows = sqlContext.createDataFrame(rowRDD,StructType(fields))

    val uri = url + "?user=" + username + "&password=" + password + "&useUnicode=true&characterEncoding=UTF-8"
    val properties = new Properties()
    properties.setProperty("user",username)
    properties.setProperty("password",password)
    //注意：集群上运行时，一定要添加这句话，否则会报找不到mysql驱动的错误
    properties.put("driver", "com.mysql.jdbc.Driver")

    rows.write.mode("append").jdbc(uri,"user_recommend",properties)
  }


}