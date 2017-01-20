# -*- coding: utf-8 -*-
print(__doc__)
"""
Created on Sat Sep 17 00:46:18 2016

@author: SROY
"""

from sklearn.datasets import load_iris
from sklearn import tree
iris=load_iris()

IndicesTrain = [38, 101, 149, 136, 113, 97, 141, 36, 37, 13, 68, 11, 52, 49, 65, 118, 55, 67, 3, 17, 135, 29, 10, 22, 2, 31, 96, 89, 27, 108, 51, 1, 16, 106, 86, 34, 104, 93, 134, 43, 111, 39, 47, 78, 116, 64, 94, 24, 132, 58, 8, 127, 71, 66, 130, 79, 105, 63, 144, 115, 32, 25, 103, 102, 30, 76, 148, 99, 57, 107, 69, 129, 0, 109, 128, 9, 28, 90, 137, 6, 15, 82, 125, 12, 33, 91, 145, 146, 35, 140, 80, 5, 53, 139, 50, 18, 133, 119, 59, 84, 100, 147, 122, 61, 110, 72, 98, 120, 112, 48, 4, 56]
IndicesTest = [92, 44, 7, 21, 95, 75, 20, 121, 26, 19, 81, 88, 143, 117, 23, 77, 138, 73, 14, 142, 123, 62, 83, 74, 42, 60, 40, 45, 87, 124, 41, 131, 70, 46, 126, 54, 85, 114]

xTrain = iris.data[IndicesTrain]
yTrain = iris.target[IndicesTrain]

xTest = iris.data[IndicesTest]
yTest = iris.target[IndicesTest]


clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
clf.fit(xTrain, yTrain)
accuracy = clf.score(xTest, yTest)
print(accuracy)
