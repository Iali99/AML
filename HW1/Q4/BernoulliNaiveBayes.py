import json
from collections import Counter
from sklearn.naive_bayes import BernoulliNB
import timeit
import numpy as np

def convert() :
    with open('train.json') as f :
        start = timeit.default_timer()
        data = json.load(f)
        ingredientList = []
        for jsonItem in data :
            for ingredient in jsonItem["ingredients"]:
                ingredientList.append(ingredient)

        c= Counter(ingredientList).most_common(3000)
        ingredientList = []
        for item in c:
            ingredientList.append(item[0])

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
        classifier = BernoulliNB()
        classifier.fit(datasetAttr,datasetCuisine)

        end = timeit.default_timer()
        print("Model Training Time : ",end - start, " Seconds")


        start = timeit.default_timer()
        i = 0
        f = open("BernoulliNB-results.csv","w")
        f.write("id,cuisine\n")
        while i< len(testID):
            f.write(str(testID[i]))
            f.write(","),
            f.write(classifier.predict(np.reshape(testData[i],(1,-1)))[0])
            f.write("\n")
            i+=1
        end = timeit.default_timer()
        print("Testing Time : ",end - start, " Seconds")
        print("Results written to BernoulliNB-results.csv file")



if __name__ == "__main__":
    convert()
