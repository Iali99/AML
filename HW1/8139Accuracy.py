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
    tree = {}
    attributes = ["fixed acidity",	"volatile acidity",	"citric acid",	"residual sugar",	"chlorides",	"free sulfur dioxide",	"total sulfur dioxide",	"density",	"pH",	"sulphates",	"alcohol"]
    usedAttr = [0,0,0,0,0,0,0,0,0,0,0]
    rootNode = None
    counter = 0
    def learn(self, training_set):
        # implement this function
        self.tree = {}

        # rootNode = self.addTreeNode(training_set,self.tree,0,self.usedAttr,0)
        self.rootNode = self.getRootNode(training_set)
        print("Total Nodes = %d" % self.counter)
        # def getKey(item) :
        #     return item[0]

        # self.tree = sorted(self.tree,key=getKey)
        # print("FINAL TREE :")
        # for x in self.tree :
        #     print(x)
        #     print(self.tree[x])

        # print("FINAL :: attr = %s splittingPoint = %f , gain = %f" % (self.attributes[attrColumn],splitPoint, gain))
        # extract attributes from dataset and evaluate the gain for each attribute
            # create an empty list of size 11 (11 attributes) to store the gain of each attribute
            # iterate over attributes and evaluate gain
                # add gain to the list

        # Select the highest gain attribute and calculate the splitting point
            # use this 'https://datascience.stackexchange.com/questions/24339/how-is-a-splitting-point-chosen-for-continuous-variables-in-decision-trees' to get splitting algorithm

        # Split the dataset into 2 parts based on splitting point

        # Find the next attribute to split on the two parts of the data set recursively

        # Repeat the above process till the max length of tree is reached



    # implement this function
    def classify(self, test_instance):
        result = 0 # baseline: always classifies as 0
        currentNode = self.rootNode
        if currentNode == None :
            print("None")
        while(currentNode.isLeaf == False):
            # print("a")
            if float(test_instance[currentNode.attrNumber]) <= currentNode.splittingPoint:
                currentNode = currentNode.leftChild
            else:
                currentNode = currentNode.rightChild
        result = currentNode.result
        # print("--------")
        return result


    # implement the getRootNode function
    def getRootNode(self,training_set):
        node = self.addTreeNode(training_set,self.tree,0,self.usedAttr,0)
        if node == None :
            print("Hey its none here")
        return node

    # implement the add node function
    def addTreeNode(self,training_set,tree,nodeNo,usedAttr,level):
        c = Counter(elem[-1] for elem in training_set)
        positives = c['1']
        negatives = c['0']
        classEntropy = self.getEntropy(positives,negatives)
        self.counter += 1
        attrColumn = -1
        splitPoint = 0
        gain = -1
        i = 0
        # thisNode = nodeNo
        while i < 11 :
            # if usedAttr[i] != 1 :
            sp,tempGain = self.getSPandGain(training_set,i,classEntropy)
            # print("tempGain = %f"% tempGain)
                # print("attr = %s splittingPoint = %f , gain = %f" % (self.attributes[i],sp, tempGain))
            if gain < tempGain :
                gain = tempGain
                attrColumn = i
                splitPoint = sp
            i += 1


        if gain == 0 :
            print("gain = %f" % gain)
            node = Node()
            node.isLeaf = True
            node.result = training_set[0][-1]
            return node
        usedAttr[attrColumn] = 1
        usedAttr1 = usedAttr[:]
        usedAttr2 = usedAttr[:]

        thisNode = Node()
        thisNode.attrNumber = attrColumn
        thisNode.splittingPoint = splitPoint

        # tup =(attrColumn,-1,-1,-1,-1,splitPoint)
        # lst = list(tup)

        leftSet = [elem for elem in training_set if float(elem[attrColumn]) <= splitPoint]
        rightSet = [elem for elem in training_set if float(elem[attrColumn]) > splitPoint]
        # print("total = %d, left = %d, right = %d" % (len(training_set),len(leftSet), len(rightSet)))
        c = Counter(elem[-1] for elem in rightSet)
        p1 = c['1']
        n1 = c['0']
        if self.getEntropy(p1,n1) == 0 :
            # lst[2] = int(1)
            # lst[4] = Counter(map(itemgetter(11), rightSet)).most_common(11)[0][0]
            # self.addTreeNode(rightSet,tree,nodeNo,usedAttr1,level+1)
            node = Node()
            node.isLeaf = True
            node.result = 1 if p1 > n1 else 0
            thisNode.rightChild = node
        else :
            # lst[2] = int(0)
            # lst[4] = self.addTreeNode(rightSet,tree,nodeNo +1,usedAttr1,level+1)
            # nodeNo = lst[4]
            node = self.addTreeNode(rightSet,tree,nodeNo +1,usedAttr1,level+1)
            thisNode.rightChild = node

        c = Counter(elem[-1] for elem in leftSet)
        p2 = c['1']
        n2 = c['0']
        # if self.getEntropy(p2,n2) == 0 or level == 10:
        #     lst[1] = int(1)
        #     lst[3] = Counter(map(itemgetter(11), leftSet)).most_common(11)[0][0]
        # else :
        #     lst[1] = int(0)
        #     lst[3] = self.addTreeNode(leftSet,tree,nodeNo+1,usedAttr2,level+1)
        #     nodeNo = lst[3]
        if self.getEntropy(p2,n2) == 0:
            # lst[2] = int(1)
            # lst[4] = Counter(map(itemgetter(11), rightSet)).most_common(11)[0][0]
            node = Node()
            node.isLeaf = True
            node.result =  1 if p2 > n2 else 0
            thisNode.leftChild = node
        else :
            # lst[2] = int(0)
            # lst[4] = self.addTreeNode(rightSet,tree,nodeNo +1,usedAttr1,level+1)
            # nodeNo = lst[4]
            node = self.addTreeNode(leftSet,tree,nodeNo +1,usedAttr1,level+1)
            thisNode.leftChild = node

        # tree[thisNode] = tuple(lst)
        if thisNode == None:
            print("Node is None")
        return thisNode

    # implement the splitting point function
    def getSPandGain(self,training_set,columnNO,classEntropy) :
        def getKey(item):
            return item[columnNO]
        training_set = sorted(training_set, key=getKey)
        gain = -1
        splittingPoint = -1
        i = 0
        while i < len(training_set) -1 :
            if training_set[i][-1] != training_set[i+1][-1] :
                sp = (float(training_set[i][columnNO]) + float(training_set[i+1][columnNO]))/2
                # print(sp)
                tempGain = self.getGain(training_set,columnNO,sp,classEntropy)
                if tempGain > gain :
                    gain = tempGain
                    splittingPoint = sp
            i += 1
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
    training_set = [x for i, x in enumerate(data) if i % K != 9]
    test_set = [x for i, x in enumerate(data) if i % K == 9]

    tree = DecisionTree()
    # Construct a tree using training set
    print("Starting to Train Model....")
    tree.learn( training_set )
    print("Model Trained. Starting Testing.........")
    # Classify the test set using the tree we just constructed
    results = []
    for instance in test_set:
        result = tree.classify( instance[:-1] )
        results.append( result == int(instance[-1]))

    # Accuracy
    accuracy = float(results.count(True))/float(len(results))
    print( "accuracy: %.4f" % accuracy)


    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()


if __name__ == "__main__":
    run_decision_tree()
