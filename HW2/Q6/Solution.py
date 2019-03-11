from DecisionTree import DecisionTree
from DecisionTree import createDecisionTree
from DecisionTree import getOOBprediction
from collections import Counter
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import time

class RandomForest():
    Forest = []

    def learn(self,training_set,m):
        for i in range(10):
            self.Forest.append(createDecisionTree(m,training_set))

    def classify(self, test_instance):
        results = []
        for tree in self.Forest:
            results.append(tree.classify(test_instance))
        freq = Counter(results)
        if freq[0] >= freq[1] :
            return 0
        else :
            return 1

    def predict(self,test_set):
        predictions = []
        for instance in test_set:
            predictions.append(self.classify(instance[:-1]))
        return predictions


def run():
    data = np.loadtxt('spam.data')
    np.random.shuffle(data)

    K = 10
    training_set = [x.tolist() for i, x in enumerate(data) if i % K >= 3 ]
    testing_set = [x.tolist() for i, x in enumerate(data) if i % K <3 ]

    print("Part A :")
    partA(training_set,testing_set,20)
    print("Part B :")
    partB(training_set,testing_set)
    print("Part C :")
    partC(training_set)

def partA(training_set,testing_set,m):

    start = time.time()
    rf = RandomForest()
    rf.learn(training_set,m)

    predictions = rf.predict(testing_set)
    predictionsArr = np.array(predictions)
    testClasses = np.array(testing_set)[:,-1]
    accuracy = np.sum(predictionsArr == testClasses)/predictionsArr.size

    print("My model Accuracy = ",accuracy*100)
    end = time.time()
    print("Time taken by my model = ", end - start)

    trainSet = np.delete(np.array(training_set),-1,axis=1)
    testSet = np.delete(np.array(testing_set),-1,axis=1)
    trainingClasses = np.array(training_set)[:,-1]
    start = time.time()
    skRF = RandomForestClassifier(n_estimators=10,criterion='entropy',max_features=m)
    skRF.fit(trainSet,trainingClasses)

    predictions = skRF.predict(testSet)
    accuracy = np.sum(predictions == testClasses) / predictions.size

    print("Sklearn Accuracy = ",accuracy)
    end = time.time()
    print("Time taken Sklearn model = ", end - start)

def partB(training_set,testing_set):
    i = 5
    while i < 55 :
        print("m = ",i)
        partA(training_set,testing_set,i)
        i += 5

def partC(training_set):
    m = 5
    while m <55:
        pred_dict = defaultdict(list)
        count = 0
        for i in range(10):
            getOOBprediction(m,training_set,pred_dict)
        for index in pred_dict:
            result = pred_dict[index]
            freq = Counter(result)
            prediction = -1
            if freq[0] >= freq[1]:
                prediction = 0
            else :
                prediction = 1
            elem = training_set[index]
            if prediction ==  elem[-1]:
                count += 1

        accuracy = count / len(pred_dict)

        print("OOB accuracy = ",accuracy*100," for m = ",m)
        m += 5

if __name__ == "__main__":
    run()
