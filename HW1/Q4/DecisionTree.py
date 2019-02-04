import json
from collections import Counter
from sklearn import tree
import timeit
import numpy as np

def run() :
    with open('train.json') as f :
        start = timeit.default_timer()
        data = json.load(f)
        ingredientList = []
        for jsonItem in data :
            for ingredient in jsonItem["ingredients"]:
                ingredientList.append(ingredient)

        c= Counter(ingredientList).most_common(3500)
        ingredientList = []
        for item in c:
            ingredientList.append(item[0])
        # print(ingredientList)

        datasetAttr = []
        datasetCuisine = []
        for jsonItem in data :
            # add the ingredients list
            tempList = []
            for item in ingredientList :
                if item in jsonItem["ingredients"]:
                    tempList.append(1)
                else:
                    tempList.append(0)
            datasetAttr.append(tempList)
            datasetCuisine.append(jsonItem["cuisine"])

        # c = Counter(item for item in datasetAttr[0])
        # print(c[1])
        end = timeit.default_timer()
        print("Training Dataset Conversion Time : ",end - start, " Seconds")

        start = timeit.default_timer()
        testData = []
        testID = []
        with open('test.json') as f :
            data = json.load(f)
            for jsonItem in data :
                tempList = []
                for item in ingredientList :
                    if item in jsonItem["ingredients"]:
                        tempList.append(1)
                    else:
                        tempList.append(0)
                testData.append(tempList)
                testID.append(jsonItem["id"])

        end = timeit.default_timer()
        print("Testing Dataset Conversion Time : ",end - start, " Seconds")

        start = timeit.default_timer()
        classifier = tree.DecisionTreeClassifier(min_impurity_split = 0.2)
        classifier.fit(datasetAttr,datasetCuisine)

        end = timeit.default_timer()
        print("Model Training Time : ",end - start, " Seconds")


        start = timeit.default_timer()
        i = 0
        # result = {}
        f = open("DecisionTree-results.csv","w")
        f.write("id,cuisine\n")
        while i< len(testID):
            f.write(str(testID[i]))
            f.write(","),
            f.write(classifier.predict(np.reshape(testData[i],(1,-1)))[0])
            f.write("\n")
            i+=1
        end = timeit.default_timer()
        print("Testing Time : ",end - start, " Seconds")
        print("Results written to DecisionTree-results.csv file")


if __name__ == "__main__":
    run()
