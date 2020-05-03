import csv
import random
from SVM import SupportVectorMachine
from SVM import (calculateInputOutput, convertSVM, classError,)
from RandomForest import Random_Forest

def crossValidationError(model="SVM"):
    """ Cross validation is calculated to analyze the prediction error in general 
        5-fold cross validation is performed the data set is divided to 5 random 
        subsets and average error is calculated"""
    if model=="SVM":
         class_data=convertSVM(list(csv.reader(open("Classified_Admission.csv","r"))))
         new_data = class_data[:]
         random.shuffle(new_data)
         error = 0
         for i in range (0,5):
             test_data = new_data[i*100:(i+1)*100]
             train_data = new_data[:]
             del train_data[i*100:(i+1)*100]
             [X,y] = calculateInputOutput(train_data)
             SVM = SupportVectorMachine()
             SVM.fit(X,y)
             [X_t,y_t] = calculateInputOutput(test_data)
             y_p = SVM.predict(X_t)
             error  += classError(y_p,y_t)
             return error/5
    # This method will run Random forest 5 times and take the average of the f1 score 
    elif model=="RandomForest":
        f1=0
        for i in range(5):
            print(i,"'th run")
            forest=Random_Forest()
            f1=f1+forest.returnf1()
        fNew=f1/5    
        print("Average f1 score of the RandomForest after 5 run: ",fNew)    
        return fNew   
    else: 
        print("EXCEPTİON")     
    
   
     