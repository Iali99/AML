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

# Implement the Tree Node Class
class Node():
    splittingPoint = -1
    attrNumber = -1
    isLeaf = False
    result = -1
    leftChild = None
    rightChild = None

# Implement your decision tree below
class DecisionTree():
    rootNode = None
    counter = 0

    def __init__(self,m):
        self.m = m

    def learn(self, training_set):
        # implement this function
        self.rootNode = self.getRootNode(training_set)\



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
        positives = c[1]
        negatives = c[0]
        classEntropy = self.getEntropy(positives,negatives)
        self.counter += 1
        attrColumn = -1
        splitPoint = 0
        gain = -1
        a = np.sort(np.random.choice(len(training_set[0])-1 ,self.m,replace=False))
        for i in a :
            sp,tempGain = self.getSPandGain(training_set,i,classEntropy)
            if gain < tempGain :
                gain = tempGain
                attrColumn = i
                splitPoint = sp

        thisNode = Node()
        thisNode.attrNumber = attrColumn
        thisNode.splittingPoint = splitPoint

        leftSet = [elem for elem in training_set if float(elem[attrColumn]) <= splitPoint]
        rightSet = [elem for elem in training_set if float(elem[attrColumn]) > splitPoint]

        c = Counter(elem[-1] for elem in rightSet)
        p1 = c[1]
        n1 = c[0]
        c = Counter(elem[-1] for elem in leftSet)
        p2 = c[1]
        n2 = c[0]

        if self.getEntropy(p1,n1) == 0 or (p1 == positives and n1 == negatives):
            node = Node()
            node.isLeaf = True
            node.result = 1 if p1 > n1 else 0
            thisNode.rightChild = node
        else :
            node = self.addTreeNode(rightSet)
            thisNode.rightChild = node


        if self.getEntropy(p2,n2) == 0 or (p2 == positives and n2 == negatives):
            node = Node()
            node.isLeaf = True
            node.result =  1 if p2 > n2 else 0
            thisNode.leftChild = node
        else :
            node = self.addTreeNode(leftSet)
            thisNode.leftChild = node

        return thisNode

    # splitting point function using mean.
    def getSPandGain(self,training_set,columnNO,classEntropy) :
        def avg(lst):
            return sum(lst)/len(lst)
        splittingPoint = avg([float(elem[columnNO]) for elem in training_set])
        gain = self.getGain(training_set,columnNO,splittingPoint,classEntropy)
        return splittingPoint, gain


    # implement the gain function
    def getGain(self,training_set,columnNO,splittingPoint,classEntropy) :
        c = Counter(elem[-1] for elem in training_set if float(elem[columnNO]) > splittingPoint)
        p1 = c[1]
        n1 = c[0]
        c = Counter(elem[-1] for elem in training_set if float(elem[columnNO]) <= splittingPoint)
        p2 = c[1]
        n2 = c[0]

        entropy = ((p1+n1)*self.getEntropy(p1,n1) + (p2+n2)*self.getEntropy(p2,n2))/len(training_set)

        return classEntropy - entropy

    # implement the entropy function
    def getEntropy(self,p,n):
        entropy = 0
        total = p + n
        if(p>0) :
            entropy -= (p/total)*math.log(p/total,2)
        if(n>0) :
            entropy -= (n/total)*math.log(n/total,2)
        return entropy


def createDecisionTree(m,training_set):

    array = np.array(training_set)
    array_bootstraped = array[np.random.choice(len(training_set),int(len(training_set)),replace=True),:]
    bootstraped_set = [elem.tolist() for elem in array_bootstraped]

    tree = DecisionTree(m)
    tree.learn( bootstraped_set )
    return tree

def getOOBprediction(m,training_set,pred_dict):

    array = np.array(training_set)
    array_bootstraped = array[np.random.choice(len(training_set),int(len(training_set)),replace=True),:]
    bootstraped_set = [elem.tolist() for elem in array_bootstraped]

    tree = DecisionTree(m)
    tree.learn( bootstraped_set )
    OOBset = [elem for elem in training_set if not(elem in bootstraped_set)]

    for elem in OOBset:
        index = training_set.index(elem)
        pred_dict[index].append(tree.classify(elem[:-1]))

    
