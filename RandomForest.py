import csv
import math
import random
import numpy as np
from DecisionTree import DecisionTree
from evaluation import f1_score


class Random_Forest(DecisionTree):
    """Random forest class which consists of Decision_Tree() class"""
    # X is the and y is class label
    def __init__(self,class_train,class_test,min_division=2,min_info_gain=0.05,number_of_trees=15):
        # Labels are appended to X_train & X_test
        self.X_train=list(csv.reader(open(class_train,'r')))
        self.X_test=list(csv.reader(open(class_test,'r')))
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
            forest.append(tree.train(self.get_random_subsets(indexes)))
        # making predictions and calculating the f1 score
        pred=self.predict(forest,self.X_test,number_of_trees)
        f1_score([row[8] for row in self.X_test],pred)
    
    # make prediction by voting
    def predict(forest,test_data,number_of_trees):
        prediction=np.zeros(test_data.shape[0])
        for i in range(len(prediction)):
            for j in range(number_of_trees):
                prediction+=forest[j].predict(test_data)
        prediction=prediction[prediction>7]    
        return prediction
    
    # random division of data samples and features with replacement
    def get_random_subsets(self,indexes,subsample_size=50):
        print(self.X_train)
        X_shuffled=list(np.random.shuffle(self.X_train))
        print(X_shuffled)
        subset=X_shuffled[1:subsample_size][:]
        subset=subset[:][indexes[0],indexes[1]]
        return subset