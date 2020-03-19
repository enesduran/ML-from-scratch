# import from other files
from data_preprocess import classifier
from data_preprocess import PCA
from data_preprocess import split_data
from RandomForest import Random_Forest
from kNN import kNN
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
# pca
PCA(normalizedFile)
# kNN algorithm
kNN(k,reg_train,reg_test,test_data_size)
# implementing random forest
Random_Forest() 
#svm()