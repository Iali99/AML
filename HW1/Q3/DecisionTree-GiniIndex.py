# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.
from __future__ import division
import csv
import math
import numpy as np
from collections import Counter
from operator import itemgetter


# Enter You Name Here
myname = "Irfan-Ali-" # or "Doe-Jane-"

# Implement the Node Class
class Node():
    splittingPoint = -1
    attrNumber = -1
    isLeaf = False
    result = -1
    leftChild = None
    rightChild = None

# Implement your decision tree below
class DecisionTree():
    attributes = ["fixed acidity",	"volatile acidity",	"citric acid",	"residual sugar",	"chlorides",	"free sulfur dioxide",	"total sulfur dioxide",	"density",	"pH",	"sulphates",	"alcohol"]
    rootNode = None
    counter = 0
    threshold = 0.11
    def learn(self, training_set):
        # implement this function
        print("threshold = %f" % self.threshold)
        self.rootNode = self.getRootNode(training_set)
        print("Total Nodes = %d" % self.counter)


    # implement this function
    def classify(self, test_instance):
        result = 0 # baseline: always classifies as 0
        currentNode = self.rootNode
        while(currentNode.isLeaf == False):
            if float(test_instance[currentNode.attrNumber]) <= currentNode.splittingPoint:
                currentNode = currentNode.leftChild
            else:
                currentNode = currentNode.rightChild
        result = currentNode.result
        return result


    # implement the getRootNode function
    def getRootNode(self,training_set):
        node = self.addTreeNode(training_set)
        return node

    # implement the add node function
    def addTreeNode(self,training_set):
        c = Counter(elem[-1] for elem in training_set)
        positives = c['1']
        negatives = c['0']
        classGI = self.getGiniIndex(positives,negatives)
        self.counter += 1
        attrColumn = -1
        splitPoint = 0
        gain = -1
        i = 0
        while i < 11 :
            sp,tempGain = self.getSPandGI(training_set,i,classGI)
            if gain < tempGain :
                gain = tempGain
                attrColumn = i
                splitPoint = sp
            i += 1

        thisNode = Node()
        thisNode.attrNumber = attrColumn
        thisNode.splittingPoint = splitPoint

        leftSet = [elem for elem in training_set if float(elem[attrColumn]) <= splitPoint]
        rightSet = [elem for elem in training_set if float(elem[attrColumn]) > splitPoint]

        c = Counter(elem[-1] for elem in rightSet)
        p1 = c['1']
        n1 = c['0']
        if self.getGiniIndex(p1,n1) <= self.threshold :
            node = Node()
            node.isLeaf = True
            node.result = 1 if p1 > n1 else 0
            thisNode.rightChild = node
        else :
            node = self.addTreeNode(rightSet)
            thisNode.rightChild = node

        c = Counter(elem[-1] for elem in leftSet)
        p2 = c['1']
        n2 = c['0']

        if self.getGiniIndex(p2,n2) <= self.threshold :
            node = Node()
            node.isLeaf = True
            node.result =  1 if p2 > n2 else 0
            thisNode.leftChild = node
        else :
            node = self.addTreeNode(leftSet)
            thisNode.leftChild = node

        return thisNode


    # splitting point function using mean.
    def getSPandGI(self,training_set,columnNO,classGI) :
        def avg(lst):
            return sum(lst)/len(lst)
        splittingPoint = avg([float(elem[columnNO]) for elem in training_set])
        gain = self.getGain(training_set,columnNO,splittingPoint,classGI)
        return splittingPoint, gain


    # implement the gain function
    def getGain(self,training_set,columnNO,splittingPoint,classGI) :
        c = Counter(elem[-1] for elem in training_set if float(elem[columnNO]) > splittingPoint)
        p1 = c['1']
        n1 = c['0']
        c = Counter(elem[-1] for elem in training_set if float(elem[columnNO]) <= splittingPoint)
        p2 = c['1']
        n2 = c['0']
        giniIndex = ((p1+n1)*self.getGiniIndex(p1,n1) + (p2+n2)*self.getGiniIndex(p2,n2))/len(training_set)

        return classGI - giniIndex

    # implement the entropy function
    def getGiniIndex(self,p,n):
        index = 0
        total = p + n
        if total > 0:
            index = 1
            index -= (p/total)*(p/total)
            index -= (n/total)*(n/total)
        return index


def run_decision_tree():

    # Load data set
    with open("wine-dataset.csv") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]
    print ("Number of records: %d" % len(data))

    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    m = 0
    results = []
    while m < 10 :
        training_set = [x for i, x in enumerate(data) if i % K != m]
        test_set = [x for i, x in enumerate(data) if i % K == m]

        tree = DecisionTree()
        # Construct a tree using training set
        print("Starting to Train Model....")
        tree.learn( training_set )
        print("Model Trained. Starting Testing.........")
        # Classify the test set using the tree we just constructed

        for instance in test_set:
            result = tree.classify( instance[:-1] )
            results.append( result == int(instance[-1]))
        m+=1

    # Accuracy
    accuracy = float(results.count(True))/float(len(results))
    print( "accuracy: %.4f" % accuracy)


    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()


if __name__ == "__main__":
    run_decision_tree()
