import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mvn

class Bayes:
    def fit(self, X, Y, k=0.01):
        self.gaussian = {}
        self.prior = {}
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussian[c] = {'mean':current_x.mean(axis=0),'variance': current_x.var(axis=0)+k}
            self.prior[c] = float(len(Y[Y == c])) / len(Y)
    def predict(self, X,Y):
        row,col = X.shape
        P = np.zeros((row, len(self.gaussian)))
        for i,j in (self.gaussian).items():
            mean = j['mean']
            var = j['variance']
            P[:,i] = mvn.logpdf(X, mean=mean, cov=var)+np.log(self.prior[i])
        alpha =np.argmax(P, axis=1)
        return np.mean(alpha == Y)

def readData(file):
    f = open(file, 'r')
    data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].replace(" ", '0')
        data[i] = data[i].replace("+", "1")
        data[i] = data[i].replace("#", "1")
    d={}
    index = 0
    for i in range(len(data)):
        if (i !=0 and i%28 ==0):
            index = index+1
        if index in d.keys():
            d[index].extend([int(i) for i in data[i] if i != '\n'])
        else:
            d[index] = [int(i) for i in data[i] if i != '\n']
    return d
def readLabels(file):
    f = open(file, "r")
    label = f.readlines()
    label = [int(i) for i in label]
    return label
trainD = readData("trainingimages")
dataFramesTrain = pd.DataFrame().from_dict(trainD, orient = "index")
trainLabels = readLabels("traininglabels")
dataFramesTrain['class'] = None
index = 0
for i in trainLabels:
    dataFramesTrain.loc[index, 'class'] = i
    index = index+1
testD = readData("testimages")
dataFramesTest = pd.DataFrame().from_dict(testD, orient='index')
dataFramesTest["class"] = None
testLabels = readLabels("testlabels")
index = 0
for i in testLabels:
    dataFramesTest.loc[index, 'class'] = i
    index = index+1
XTest = dataFramesTest.drop("class", axis =1)
yTest = dataFramesTest['class']
XTrain = dataFramesTrain.drop("class", axis = 1)
yTrain = dataFramesTrain['class']
model = Bayes()
model.fit(XTrain, yTrain)
print("Train Accuracy: ",model.predict(XTrain,yTrain ))
print("Test Accuracy: ", model.predict(XTest, yTest))
#function prints the confusion matrix.
from sklearn.metrics import confusion_matrix
def prediction(X):
    row,col = X.shape
    P = np.zeros((row, len(model.gaussian)))
    for x, y in (model.gaussian).items():
        mean, var = y['mean'], y['variance']
        P[:,x] = mvn.logpdf(X, mean=mean, cov=var) + np.log(model.prior[x])
    return np.argmax(P, axis=1)
x = prediction(XTest)
cm = confusion_matrix(yTest, x)
print(cm)


