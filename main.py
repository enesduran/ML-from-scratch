import csv
import numpy as np
# import from other files
from data_preprocess import classifier
from data_preprocess import PCA
from data_preprocess import split_data
from data_preprocess import calculateInputOutput
from data_preprocess import convertSVM
from RandomForest import Random_Forest
from kNN import kNN
from SVM import SupportVectorMachine
from SVM import classError 
from SVM import crossValidationError


# test dataset volume
test_data_size=100
# threshold for classification
threshold=0.7
k=8
# dataset taken from Kaggle
fileName='Admission_Predict.csv'
# normalized version of it
normalizedFile="Regression_Admission.csv"
# dataset directories
class_test="Classification_Test_Data.csv"
reg_test="Regression_Test_Data.csv"
# train sets
reg_train="Regression_Train.csv"
class_train="Classification_Train.csv"
# preparing the data
classifier(threshold,fileName,normalizedFile)
split_data(test_data_size,reg_train,class_train,reg_test,class_test)
# dividing the train data as input and output for SVM algorithm
#[X, y] = calculateInputOutput(convertSVM(list(csv.reader(open(class_train,'r')))))
# creating SVM object from SupportVectorMachine class
#SVM = SupportVectorMachine()
# fitting the train data to the SVM machine
#SVM.fit(X, y)
# dividing the test data as input and output for SVM algorithm
#[X_t, y_t] = calculateInputOutput(convertSVM(list(csv.reader(open(class_test,'r')))))
# classifying the test data with the SVM machine
#y_p = SVM.predict(X_t)
#error  = classError(y_p,y_t)
#cross_validation_error = crossValidationError()

# # pca
PCA(normalizedFile)
# # kNN algorithm
kNN(k,reg_train,reg_test,test_data_size)
# # implementing random forest
Random_Forest(class_train,class_test) 
