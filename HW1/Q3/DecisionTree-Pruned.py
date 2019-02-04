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
    threshold = 0.32
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
        classEntropy = self.getEntropy(positives,negatives)
        self.counter += 1
        attrColumn = -1
        splitPoint = 0
        gain = -1
        i = 0
        while i < 11 :
            sp,tempGain = self.getSPandGain(training_set,i,classEntropy)
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
        if self.getEntropy(p1,n1) <= self.threshold :
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

        if self.getEntropy(p2,n2) <= self.threshold :
            node = Node()
            node.isLeaf = True
            node.result =  1 if p2 > n2 else 0
            thisNode.leftChild = node
        else :
            node = self.addTreeNode(leftSet)
            thisNode.leftChild = node

        return thisNode

    # implement the splitting point function using a better strategy
    # def getSPandGain(self,training_set,columnNO,classEntropy) :
    #     def getKey(item):
    #         return item[columnNO]
    #     training_set = sorted(training_set, key=getKey)
    #     gain = -1
    #     splittingPoint = -1
    #     i = 0
    #     while i < len(training_set) -1 :
    #         if training_set[i][-1] != training_set[i+1][-1] :
    #             sp = (float(training_set[i][columnNO]) + float(training_set[i+1][columnNO]))/2
    #             tempGain = self.getGain(training_set,columnNO,sp,classEntropy)
    #             if tempGain > gain :
    #                 gain = tempGain
    #                 splittingPoint = sp
    #         i += 1
    #     return splittingPoint, gain

    # splitting point function using mean. Use the below function by commenting the above function and uncommenting this one
    def getSPandGain(self,training_set,columnNO,classEntropy) :
        def avg(lst):
            return sum(lst)/len(lst)
        splittingPoint = avg([float(elem[columnNO]) for elem in training_set])
        gain = self.getGain(training_set,columnNO,splittingPoint,classEntropy)
        return splittingPoint, gain


    # implement the gain function
    def getGain(self,training_set,columnNO,splittingPoint,classEntropy) :
        c = Counter(elem[-1] for elem in training_set if float(elem[columnNO]) > splittingPoint)
        p1 = c['1']
        n1 = c['0']
        c = Counter(elem[-1] for elem in training_set if float(elem[columnNO]) <= splittingPoint)
        p2 = c['1']
        n2 = c['0']

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
