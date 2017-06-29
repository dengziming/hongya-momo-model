package com.hongya.bigdata.day05

import java.io.PrintWriter

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.Map

/**
  * Created by root on 2016/4/1 0001.
  */

object Recommend{
  def main(args: Array[String]) {

    val conf = new SparkConf().setMaster("local").setAppName("LRSGD")
    val sc =  new SparkContext(conf)
    //加载数据，按照\t拆分，前面是标签后面是特征
    val data: RDD[Array[String]] = sc.textFile("/Users/dengziming/Documents/hongya/5.数据建模/data/traindata/000000_0").map(_.split("\t"))
    //去掉标签，统计所有特征并去重
    val trainData: RDD[String] = data.map(_.drop(1)).flatMap(x=>{
      val features: Array[String] = x(0).split(";")
      features
    }).map(_.split(":").take(1)(0)).distinct()
    //对所有特征做编号，这个顺序编号，用来确定稀疏向量的非零元素的下标
    val dict: Map[String, Long] = trainData.zipWithIndex().collectAsMap()

    val train: RDD[LabeledPoint] = data.map(x=>{
      //取出label，因为逻辑回归只支持1.0和0.0，这个一个简单的转换
      val label: Double = x(0) match {
        case "1" => 1.0
        case "-1" => 0.0
      }
      //处理一下格式，得到每个特征值字符串
      val feature: Array[String] = x.drop(1)(0).split(";").map(_.split(":").take(1)(0))
      //得到该样本非零元素的下标，实际就是从字典映射表里面取的
      val indices: Array[Int] = feature.map(a=>{
        val index: Long = dict.get(a) match{
          case Some(m) => m
          case None => 0
        }
        index.toInt
      })
      //这里是所有非零元素的值，业务定义，有值就是1其余都是0
      val values: Array[Double] = Array.fill(feature.length)(1.0)
      new LabeledPoint(label,new SparseVector(dict.size,indices,values))
    })
    //模型训练，参数分别是训练集，迭代次数和步长
    val model: LogisticRegressionModel = LogisticRegressionWithSGD.train(train,10,0.1)
    val weights: Array[Double] = model.weights.toArray
    //字典表反转，为了通过下标找出对应的特征
    val map: Map[Long, String] = dict.map(x=>{(x._2,x._1)})
    val pw = new PrintWriter(new java.io.File("F:\\hongya\\data\\model\\out"))
    for(i <- weights.indices){
      //通过下标找出特征，相应的下标的权重和相应下标的特征字符串对应起来了
      val fea: String = map.get(i) match {
        case Some(x) => x
        case None => ""
      }
      val result = fea+"\t"+weights(i)
      pw.write(result)
      pw.println()
      println(result)
    }
    pw.flush()
    pw.close()

  }
}
