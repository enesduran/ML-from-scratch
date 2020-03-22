import numpy as np
import math

class DecisionTree():
    
    def __init__(self,feature_1,feature_2,min_info_gain,min_number_of_division):
        self.feature_1=feature_1
        self.feature_2=feature_2
        self.min_info_gain=min_info_gain
        self.min_number_of_division=min_number_of_division
    
    def split(self):
        y,y1,y2=0
        if self.information_gain(y,y1,y2)>self.min_info_gain:
            pass
        return
    
    def train(X,y):
        return
   
    def test(x):
        result=0
        return result
    
    # calculate info gain 
    def information_gain(self,y,y1,y2):
        prob=len(y1)/len(y)
        info_gain=self.entropy(y)-self.entropy(y1)*prob+self.entropy(y2)*(1-prob)
        return info_gain
    
    # entropy calculation of a node
    def entropy(y):
        # there will always be 2 classes at max
        entropy=0
        # determining number of classes
        if (0 in y) and (1 in y):
            class_number=2
        else:
            class_number=1 
        for i in class_number:
            probability=len(y[y == i])/len(y)
            entropy+= probability*math.log(probability,2)
        return entropy
    
    # return the predictions in an array 
    def predict(X_test):
        [m,n]=X_test.shape
        y_predict=np.zeros(m)
        # iterating over data points
        idx=[i for i in m]
        y_predict=y_predict[X_test[idx]==1]
#         for i in range(len(y_predict)):
#             if test(X_test[i])==1:
        return y_predict

