from sklearn import svm
import numpy as np

def getData(filename):
    completeSet = np.loadtxt(filename)
    filteredSet = completeSet[(completeSet[:,0] == 1) | (completeSet[:,0] == 5)]
    attrs = filteredSet[:,[1,2]]
    classes = filteredSet[:,[0]].ravel()
    return attrs,classes

def run():
    trainAttrs, trainClasses = getData('features.train')
    testAttrs, testClasses = getData('features.test')

    partA(trainAttrs, trainClasses, testAttrs, testClasses)

    B = [50,100,200,800]
    for n in B:
        partB(trainAttrs, trainClasses, testAttrs, testClasses,n)

    C = [0.0001,0.001,0.01,1]
    for c in C:
        partC(trainAttrs, trainClasses, testAttrs, testClasses,2,c)
        partC(trainAttrs, trainClasses, testAttrs, testClasses,5,c)

    D = [0.01,1,100,10000,1000000]
    for c in D:
        partD(trainAttrs, trainClasses, testAttrs, testClasses,c)




def partA(trainAttrs, trainClasses, testAttrs, testClasses):
    linear_svc = svm.SVC(kernel='linear')
    linear_svc.fit(trainAttrs,trainClasses)

    predictions = linear_svc.predict(testAttrs)

    accuracy = np.sum(predictions == testClasses) / predictions.size

    print("Linear SVC : Number of Support Vectors = ",linear_svc.n_support_," Accuracy = ",accuracy*100)

def partB(trainAttrs, trainClasses, testAttrs, testClasses,n):
    trainAttrs = trainAttrs[:n]
    trainClasses = trainClasses[:n]

    linear_svc = svm.SVC(kernel='linear')
    linear_svc.fit(trainAttrs,trainClasses)

    predictions = linear_svc.predict(testAttrs)

    accuracy = np.sum(predictions == testClasses) / predictions.size

    print("Linear SVC : n = ",n," - Number of support vectors = ", linear_svc.n_support_," and Accuracy = ",accuracy*100)

def partC(trainAttrs, trainClasses, testAttrs, testClasses, d, c):
    poly_svc = svm.SVC(kernel = 'poly', degree = d, coef0 = 1, gamma = 'auto', C = c)
    poly_svc.fit(trainAttrs, trainClasses)

    predictions = poly_svc.predict(trainAttrs)

    accuracy = np.sum(predictions == trainClasses) / predictions.size

    print("Polynomial SVC : Training Error for degree = ",d," and C = ",c," = ",(1-accuracy)*100)
    predictions = poly_svc.predict(testAttrs)

    accuracy = np.sum(predictions == testClasses) / predictions.size

    print("Polynomial SVC : Testing Error for degree = ",d," and C = ",c," = ",(1-accuracy)*100)
    print("Number of Support Vectors = ",poly_svc.n_support_)

def partD(trainAttrs, trainClasses, testAttrs, testClasses, c):
    rbf_svc = svm.SVC(kernel = 'rbf', gamma = 'auto', C = c)
    rbf_svc.fit(trainAttrs,trainClasses)

    predictions = rbf_svc.predict(trainAttrs)

    accuracy = np.sum(predictions == trainClasses) / predictions.size
    print("RBF SVC : Training error for C = ",c," = ",(1-accuracy)*100)

    predictions = rbf_svc.predict(testAttrs)

    accuracy = np.sum(predictions == testClasses) / predictions.size

    print("RBF SVC : Testing error for C = ",c," = ",(1-accuracy)*100)

if __name__ == "__main__":
    run()
