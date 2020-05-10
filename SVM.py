import csv
import numpy as np
import random
from data_preprocess import calculateInputOutput
from data_preprocess import convertSVM
from evaluation import f1_score


# the main method of support vector machine called by main method
def SVM_machine(class_train,class_test):
    # dividing the train data as input and output for SVM algorithm
    [X, y] = calculateInputOutput(convertSVM(list(csv.reader(open(class_train,'r')))))
    # creating SVM object from SupportVectorMachine class
    SVM = SupportVectorMachine()
    # fitting the train data to the SVM machine
    # calculating the 5 fold cross validation
    SVM.crossValidation()
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
    score = f1_score(y_t,y_p)
    print('The SVM f1 score is {0:.2f} \n'.format(score))
    return

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
        self.beta = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        return
    # minimizes the hing loss function and finds the hyperplane parameters 
    # stochastic gradient descent is applied to find the minimum of the loss 
    def fit(self,X,y):
        n_samples,n_features = np.shape(X)
        #sgd step size
        gama = 0.01
        t = 0
        #sgd batch size
        m=150
        #randomizing the sgd samples
        a = np.arange(400)
        np.random.shuffle(a)
        for p in range(m):
            grad = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            j = a[p]
            t = np.dot(X[j],self.beta[0:len(self.beta)-1]) + self.beta[len(self.beta)-1]
            if (t * y[j]) < 1 :
                for i in range(len(grad)-1):
                    grad[i] -= X[j][i] * y[j]
                grad[len(grad)-1] -= y[j]
                for k in range(len(self.beta)):
                    self.beta[k] = self.beta[k] - gama * grad[k]
        return
            
    # the cross validation is calculated to find the optimal svm paramters 
    # 5-fold cross validation is performed
    # the data set is divided to 5 random subsets and average error is calculated     
    def crossValidation(self):
        class_data=convertSVM(list(csv.reader(open("Classified_Admission.csv","r"))))
        new_data = class_data[:]
        random.shuffle(new_data)
        for i in range (0,5):
            train_data = new_data[:]
            del train_data[i*100:(i+1)*100]
            [X,y] = calculateInputOutput(train_data)
            self.fit(X,y)
        print("betas are:")
        print(self.beta)
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
  
        
        
        