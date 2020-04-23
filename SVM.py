import csv
import numpy as np
import random
from data_preprocess import calculateInputOutput
from data_preprocess import convertSVM
#for optimization import cvxopt library 
#import cvxopt

# the kernel type for linear hyperplane
def linear_kernel(x1,x2):
    return np.dot(x1,x2)
# the kernel type for polynomial hyperplane
def polynomial_kernel(x1,x2,p):
    return (1+np.dot(x1,x2))**p
# the error of the test data classification 
def classError(y1,y2):
    missNum = 0
    for i in range (0,len(y1)-1):
        if y1[i] != y2[i]:
            missNum += 1
    return missNum/len(y1)
# the cross validation is calculated to analyze the prediction error in general 
# 5-fold cross validation is performed
# the data set is divided to 5 random subsets and average error is calculated     
def crossValidationError():
    class_data=convertSVM(list(csv.reader(open("Classified_Admission.csv","r"))))
    new_data = class_data[:]
    random.shuffle(new_data)
    error = 0
    for i in range (0,5):
        test_data = class_data[i*100:(i+1)*100]
        train_data = class_data[:]
        del train_data[i*100:(i+1)*100]
        [X,y] = calculateInputOutput(train_data)
        SVM = SupportVectorMachine()
        SVM.fit(X,y)
        [X_t,y_t] = calculateInputOutput(test_data)
        y_p = SVM.predict(X_t)
        error  += classError(y_p,y_t)
    return error/5
        
# defining the SupportVectorMachine class   
# the aim of this algorithm is to maximize the margin by allowing some points
# on the wrong side of the hyperplane       
class SupportVectorMachine(object):
    # initializing the class with kernel type and constraint C(cost parameter)
    def __init__(self, kernel_type=linear_kernel, C=1):
        self.C = C
        self.kernel = kernel_type
        return
    # the fit method constructs the support vector machine using the train data
    def fit(self,X,y):
        n_samples,n_features = np.shape(X)
        # kernel matrix is build
        K = np.zeros((n_samples,n_samples));
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i],X[j])
        # convex optimization is perfomed 
        P = cvxopt.matrix(np.outer(y,y)*K, tc='d')
        self.P = P
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples), tc='d')
        b = cvxopt.matrix(0.0, tc='d')
                
        if self.C is None:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            s1 = np.diag(np.ones(n_samples) * -1)
            s2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((s1, s2)))
            s3 = np.zeros(n_samples)
            s4 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((s3, s4)))
        
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alfas = np.ravel(solution['x'])
        # the nonzero lagrange multipliers(alfas) are our support vectors(points)  
        index = alfas > 1e-6
        # the supporting points are stored
        # the alfas,the suport vectors and labeles of the supporting points     
        self.lag_multipliers = alfas[index]
        self.support_vectors = X[index]
        self.support_labels = y[index]
        
        # average of the parameter of the hyperplane is calculted to make the 
        # classficiation more stable
        self.b = 0
        for j in range(len(self.lag_multipliers)):
            self.b += self.support_labels[j]
            for i in range(len(self.lag_multipliers)):
                self.b -= self.lag_multipliers[i]*self.support_labels[i]* self.kernel(self.support_vectors[i],self.support_vectors[j])
        self.b = self.b/len(self.lag_multipliers)
            
        return
    
    # classification of the test data is performed 
    def predict(self,X):
        y_p = []
        for sample in X:
            prediction = 0
            for i in range(len(self.lag_multipliers)):
                prediction +=self.lag_multipliers[i]*self.support_labels[i]*self.kernel(sample,self.support_vectors[i])
            prediction += self.b
            y_p.append(np.sign(prediction))
        y_p=np.array(y_p,dtype=float)
        return y_p
    # checking wheter positive definite or not         
    def is_pos_def(self):
        return np.all(np.linalg.eigvals(self.P) > 0)