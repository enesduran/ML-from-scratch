import csv
import numpy as np
from data_preprocess import calculateInputOutput
from data_preprocess import convertSVM
from scipy.optimize import minimize
from evaluation import f1_score

# the main method of support vector machine called by main method
def SVM_machine(class_train,class_test):
    # dividing the train data as input and output for SVM algorithm
    [X, y] = calculateInputOutput(convertSVM(list(csv.reader(open(class_train,'r')))))
    # creating SVM object from SupportVectorMachine class
    SVM = SupportVectorMachine()
    # fitting the train data to the SVM machine
    SVM.fit(X, y)
    # dividing the test data as input and output for SVM algorithm
    [X_t, y_t] = calculateInputOutput(convertSVM(list(csv.reader(open(class_test,'r')))))
    # classifying the test data with the SVM machine
    y_p = SVM.predict(X_t)
    # printing the label predictions
    print('The SVM predictions are:')
    print(np.array(y_p))  
    #calculating the prediction error on the test set and printing 
    error  = classError(y_p,y_t)
    print('The SVM prediction error percentage is {0:.2f} \n'.format(error))
    # calculating the 5 fold cross validation error takes a little time 
    cross_validation_error = cross_val.crossValidationError()
    print('The SVM cross validation prediction error is {0:.2f} \n'.format(cross_validation_error))
    score = f1_score(y_t,y_p)
    print('The SVM f1 score is {0:.2f} \n'.format(score))

# the kernel type for linear hyperplane
def linear_kernel(x1,x2):
    return np.dot(x1,x2)

# the kernel type for polynomial hyperplane
def polynomial_kernel(x1,x2,p):
    return (1+np.dot(x1,x2))**p
     
# defining the SupportVectorMachine class   
# the aim of this algorithm is to maximize the margin by allowing some points
# on the wrong side of the hyperplane       
class SupportVectorMachine(object):
    # initializing the class with kernel type and constraint C(cost parameter)
    def __init__(self, kernel_type=linear_kernel):
        self.kernel = kernel_type
        return
    
    # calcuates the hing loss of all the data samples in the train set  
    def hinge_loss(self,b):      
        loss = 0
        n_samples,n_features = np.shape(self.X)
        for j in range(n_samples):
            t = self.kernel(self.X[j],b[0:len(b)-1]) + b[len(b)-1]
            loss += max(0,1-self.y[j]*t) 
        return loss    
    
    # minimizes the hing loss function and finds the hyperplane parameters 
    def fit(self,X,y):
        self.y= y
        self.X = X
        y0 = np.array([1,1,1,1,1,1,1,1])
        res = minimize(self.hinge_loss, y0, method='nelder-mead')
        self.beta = res.x
        return    
    
    # classification of the test data is performed 
    def predict(self,Xt):
        y_p = []
        for sample in Xt:
            prediction = self.kernel(sample,self.beta[0:len(self.beta)-1])
            prediction += self.beta[len(self.beta)-1]
            y_p.append(np.sign(prediction))
        y_p=np.array(y_p,dtype=float)
        return y_p
      
# the error of the test data classification 
def classError(y1,y2):
    missNum = 0
    for i in range (0,len(y1)-1):
        if y1[i] != y2[i]:
            missNum += 1
    return missNum/len(y1)