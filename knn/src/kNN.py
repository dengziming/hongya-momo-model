# coding=utf-8
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir


'''
四个输入:
inX 输入用户
dataSet 用于训练的样本集
labels 用户id向量
k 选择邻居的数目
'''


def nearest(user, dataSet, labels, k):
    # 得到训练集的长度
    dataSetSize = dataSet.shape[0]

    # 计算输入向量和数据集的每一个数据的差，每个差向量放进一个
    diffMat = tile(user, (dataSetSize, 1)) - dataSet

    ## 两部计算平方和
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)

    # 根号求距离
    distances = sqDistances ** 0.5

    # 按照距离排序的序号
    sortedDistIndicies = distances.argsort()

    # 得到距离相差最小的k个元素，计算每个类别的类别数
    classCount = set()
    for i in range(k):
        # 第i个的标签
        classCount.add(labels[sortedDistIndicies[i]])


    # 返回次数最多的class的标签
    return classCount


'''
文件转化为矩阵
'''
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = zeros((numberOfLines, 2))  # prepare matrix to return
    ids = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        if (listFromLine[1] == '男'):
            returnMat[index, 0] = listFromLine[1] == 1
        else:
            returnMat[index, 0] = listFromLine[1] == 0

        returnMat[index, 1] = listFromLine[2]
        ids.append(int(listFromLine[0]))
        index += 1
    return returnMat, ids


returnMat, ids = file2matrix("/Users/dengziming/Documents/hongya/data/day07/userlist.txt")
user = (1, 15)
classCount = nearest(user, returnMat, ids, 10)

print classCount