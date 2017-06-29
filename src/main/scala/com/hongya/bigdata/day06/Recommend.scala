package com.hongya.bigdata.day06

/**
  * Created by allanhwang on 17-5-9.
  */
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd._
object Recommend {

  val data_path = "/Users/dengziming/Documents/hongya/data/day06/data.txt"
  val item_path = "/Users/dengziming/Documents/hongya/data/day06/item.txt"

  def main(args: Array[String]) {
    //设置不要显示多余信息
    SetLogger
    println("==========数据准备阶段===============")
    val (ratings, movieTitle) = PrepareData()

    println("==========训练阶段===============")
    print("开始使用 " + ratings.count() + "条评比数据进行训练模型... ")
    val model = ALS.train(ratings, 5, 20, 0.1)

    println("训练完成!")
    println("==========推荐阶段===============")
    recommend(model, movieTitle)
    println("完成")
  }

  def recommend(model: MatrixFactorizationModel, movieTitle: Map[Int, String]) = {
    var choose = ""
    while (choose != "3") { //如果选择3.离开,就结束运行程序
      print("请选择要推荐类型  1.针对用户推荐电影 2.针对电影推荐感兴趣的用户 3.离开?")
      choose = readLine() //读取用户输入
      if (choose == "1") { //如果输入1.针对用户推荐电影
        print("请输入用户id?")
        val inputUserID = readLine() //读取用户ID
        RecommendMovies(model, movieTitle, inputUserID.toInt) //针对此用户推荐电影
      } else if (choose == "2") { //如果输入2.针对电影推荐感兴趣的用户
        print("请输入电影的 id?")
        val inputMovieID = readLine() //读取MovieID
        RecommendUsers(model, movieTitle, inputMovieID.toInt) //针对此电影推荐用户
      }
    }
  }


  def SetLogger = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }

  def PrepareData(): (RDD[Rating], Map[Int, String]) = {

    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[4]"))
    //----------------------1.创建用户评分数据-------------
    print("开始读取用户评分数据中...")

    val rawUserData = sc.textFile(data_path)
    val rawRatings = rawUserData.map(_.split("\t").take(3))
    val ratingsRDD = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    println("共计：" + ratingsRDD.count + "条ratings")
    //----------------------2.创建电影ID与名称对照表-------------
    print("开始读取电影数据中...")
    //val itemRDD = sc.textFile(new File(DataDir, "u.item").toString)
    //val itemRDD = sc.textFile("file:/home/hduser/workspace/Recommend/data/u.item")
    val itemRDD = sc.textFile(item_path)
    val movieTitle = itemRDD.map(line => line.split("\\|").take(2))
      .map(array => (array(0).toInt, array(1))).collect().toMap
    //----------------------3.显示数据记录数-------------
    val numRatings = ratingsRDD.count()
    val numUsers = ratingsRDD.map(_.user).distinct().count()
    val numMovies = ratingsRDD.map(_.product).distinct().count()
    println("共计：ratings: " + numRatings + " User " + numUsers + " Movie " + numMovies)
    (ratingsRDD, movieTitle)
  }

  def RecommendMovies(model: MatrixFactorizationModel, movieTitle: Map[Int, String], inputUserID: Int) = {
    val RecommendMovie = model.recommendProducts(inputUserID, 10)
    var i = 1
    println("针对用户id" + inputUserID + "推荐下列电影:")
    RecommendMovie.foreach { r =>
      println(i + "." + movieTitle(r.product) + "评分:" + r.rating.toString())
      i += 1
    }
  }

  def RecommendUsers(model: MatrixFactorizationModel, movieTitle: Map[Int, String], inputMovieID: Int) = {
    val RecommendUser = model.recommendUsers(inputMovieID, 10)
    var i = 1
    println("针对电影 id" + inputMovieID + "电影名:" + movieTitle(inputMovieID.toInt) + "推荐下列用户id:")
    RecommendUser.foreach { r =>
      println(i.toString + "用户id:" + r.user + "   评分:" + r.rating)
      i = i + 1
    }
  }

}