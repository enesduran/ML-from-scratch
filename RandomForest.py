import csv
import math
import random
import numpy as np
from DecisionTree import DecisionTree


class Random_Forest(DecisionTree):
    """Random forest class which consists of Decision_Tree() class"""
    # X is the and y is class label
    def __init__(self,class_train,class_test,min_division=2,min_info_gain=0.05,number_of_trees=15):
        # Labels are appended to X_train & X_test
        self.X_train=np.array(csv.reader(open(class_train,'r')))
        self.X_test=np.array(csv.reader(open(class_test,'r')))
        # parameter initialization
        self.number_of_trees=number_of_trees
        self.min_info_gain=min_info_gain
        self.min_division=min_division
        # feature names
        feature_names=['GRE','TOEFL','University Rating','SOP','LOR','CGPA','Research']
        feature_number_for_each_tree=math.ceil(math.sqrt(len(feature_names)))
        forest=[]
        [feature_1,feature_2]=[0,0]
        
        # initialization of decision trees 
        for i in range(number_of_trees):
            indexes=random.sample(range(len(feature_names)), feature_number_for_each_tree)
            feature_1=feature_names[indexes[0]]
            feature_2=feature_names[indexes[1]]
            tree=DecisionTree(feature_1,feature_2,min_info_gain,min_division)
            forest.append(tree.train(self.get_random_subsets()))
          
    def split_data():
        
        return
    
    def information_gain(self,y,y1,y2,min_info_gain):
        prob=len(y1)/len(y)
        info_gain=self.entropy(y)-self.entropy(y1)*prob+self.entropy(y2)*(1-prob)
        if info_gain > min_info_gain:
            self.split_data()
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
 
    # make prediction by voting
    def predict(forest,test_data,number_of_trees):
        prediction=np.zeros(test_data.shape[0])
        for i in range(len(prediction)):
            for j in range(number_of_trees):
                tree=forest[j]
                tree
        return 
    
    # random division of data samples and features with replacement
    def get_random_subsets(self,subsample_size=50):
        # number of samples 
        n_samples = np.shape(self.X_train)[0]
        X_shuffled=np.random.shuffle(self.X_train)
        for i in range(subsample_size):
            idx = np.random.choice(range(n_samples),size=np.shape(range(subsample_size)))
            X = Xy[idx][:, :-1]
            y = Xy[idx][:, -1]
            subset.append([X, y])
        return subset