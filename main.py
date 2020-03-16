import numpy as np
import csv
# import from other files
from classification import classifier
from classification import split_data
from kNN import kNN
# test dataset volume
test_data_size=100
# threshold for classification
threshold=0.7
k=10
# dataset directories
class_test="Classification_Test_Data.csv"
reg_test="Regression_Test_Data.csv"
# train sets
reg_train="Regression_Train.csv"
class_train="Classification_Train.csv"
#
class_test_data=csv.writer(open(class_test,"w",newline=""))
reg_test_data=csv.writer(open(reg_test,"w",newline=""))
reg_train_data=csv.writer(open(reg_train,"w",newline=""))
class_train_data=csv.writer(open(class_train,"w",newline="")) 
# preparing the data
classifier(threshold)
split_data(test_data_size,reg_train_data,class_train_data,reg_test_data,class_test_data)
# changing csv file writer to reader
reg_train_data=csv.reader(open(reg_train,"r"))
class_train_data=csv.writer(open(class_train,"r")) 
# kNN algorithm
kNN(k,reg_train_data,reg_test_data,test_data_size)
# decision tree algorithm 