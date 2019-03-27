import numpy as np
from numpy import genfromtxt

TrainDataPath = 'C:/testData/trainDataGradientDescent.csv' 
TestDataPath = 'C:/testData/testDataGradientDescent.csv'
TranDataSet = genfromtxt(TrainDataPath, delimiter=',')
TestDataSet = genfromtxt(TestDataPath, delimiter=',')

def getData(DataSet):
    m,n = np.shape(DataSet)
    TrainData = np.ones((m,n))
    TrainData[:,:-1] = DataSet[:,:-1]
    print("The shape of the dataset matrix is {0}*{1}".format(m,n))
    print("The former matrix: is {}".format(TrainData))
    #TrainData.shape = 3, -1
    #print("The new matrix: is {}".format(TrainData))
    

getData(TranDataSet)
