# import from other files
from classification import classifier
from classification import split_data
from kNN import kNN
from classification import PCA
# test dataset volume
test_data_size=100
# threshold for classification
threshold=0.7
k=8
# dataset taken from Kaggle
fileName='Admission_Predict.csv'
# dataset directories
class_test="Classification_Test_Data.csv"
reg_test="Regression_Test_Data.csv"
# train sets
reg_train="Regression_Train.csv"
class_train="Classification_Train.csv"
# preparing the data
classifier(threshold,fileName)
split_data(test_data_size,reg_train,class_train,reg_test,class_test)
# pca
PCA(fileName)
# kNN algorithm
kNN(k,reg_train,reg_test,test_data_size)
# decision_tree() 
#svm()