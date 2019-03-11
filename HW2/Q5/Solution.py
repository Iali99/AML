from sklearn import svm
import numpy as np

def run():
    trainAttrs = np.loadtxt('Train.data')
    trainClasses = np.loadtxt('Train.labels')

    testAttrs = np.loadtxt('Valid.data')
    testClasses = np.loadtxt('Valid.labels')

    partA(trainAttrs, trainClasses, testAttrs, testClasses)
    partB(trainAttrs, trainClasses, testAttrs, testClasses)

def partA(trainAttrs, trainClasses, testAttrs, testClasses):
    linear_svc = svm.SVC(kernel='linear')
    linear_svc.fit(trainAttrs, trainClasses)

    predictions = linear_svc.predict(testAttrs)

    accuracy = np.sum(predictions == testClasses) / predictions.size

    print("Linear SVC : Number of Support Vectors = ",linear_svc.n_support_," Testing Error = ",(1-accuracy)*100)
    predictions = linear_svc.predict(trainAttrs)

    accuracy = np.sum(predictions == trainClasses) / predictions.size

    print("Linear SVC : Testing Error = ",(1-accuracy)*100)



def partB(trainAttrs, trainClasses, testAttrs, testClasses):
    rbf_svc = svm.SVC(kernel='rbf',gamma = 0.001)
    rbf_svc.fit(trainAttrs,trainClasses)

    predictions = rbf_svc.predict(testAttrs)

    accuracy = np.sum(predictions == testClasses) / predictions.size

    print("RBF SVC : Number of Support Vectors = ",rbf_svc.n_support_," Testing Error = ",(1-accuracy)*100)

    predictions = rbf_svc.predict(trainAttrs)

    accuracy = np.sum(predictions == trainClasses) / predictions.size
    print("RBF SVC : Training error  = ",(1-accuracy)*100)

    poly_svc = svm.SVC(kernel='poly',degree = 2, coef0 = 1,gamma='auto')
    poly_svc.fit(trainAttrs,trainClasses)

    predictions = poly_svc.predict(testAttrs)

    accuracy = np.sum(predictions == testClasses) / predictions.size

    print("Polynomial SVC : Number of Support Vectors = ",poly_svc.n_support_," Testing Error = ",(1-accuracy)*100)

    predictions = poly_svc.predict(trainAttrs)

    accuracy = np.sum(predictions == trainClasses) / predictions.size
    print("Polynomial SVC : Training error  = ",(1-accuracy)*100)

if __name__ == "__main__":
    run()
