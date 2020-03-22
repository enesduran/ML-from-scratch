import numpy as np

class DecisionTree():
    
    def __init__(self,feature_1,feature_2,min_info_gain,min_number_of_division):
        self.feature_1=feature_1
        self.feature_2=feature_2
        self.min_info_gain=min_info_gain
        self.min_number_of_division=min_number_of_division
    
    def split():
        return
    
    def train(X,y):
        return
   
    def test(x):
        result=0
        return result
    
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

