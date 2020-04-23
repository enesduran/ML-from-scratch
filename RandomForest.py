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
        feature_number_for_each_tree=math.floor(math.sqrt(7))
        forest=[]
        # initialization of decision trees 
        for i in range(number_of_trees):
            feature_indexes=random.sample(range(1,7),feature_number_for_each_tree)
            tree=DecisionTree(self.get_random_subsets(feature_indexes),feature_indexes[0],feature_indexes[1],min_info_gain,min_division)
            # Add tree instance to forest
            forest.append(tree)
        # making predictions and calculating the f1 score
        pred=self.prediction(forest,number_of_trees)
        f1_score([row[8] for row in self.X_test],pred)
    
    # make prediction by voting
    def prediction(self,forest,number_of_trees):
        predictions=np.zeros(len(self.X_test))
        for j in range(number_of_trees):
            # elementwise addition
            jth_predictior=forest[j].predictionArray(self.X_test)
            predictions=[predictions[i]+jth_predictior[i] for i in range(len(predictions))] 
            predictions=[int(predictions[i]>7) for i in range(len(predictions))]    
        return predictions
    
    # random division of data samples and features with replacement
    def get_random_subsets(self,indexes,subsample_size=51):
        X_shuffled=random.sample(self.X_train,len(self.X_train))        
        temp=X_shuffled[1:subsample_size][:]
        # The label will be included
        Xy=np.array([[row[indexes[1]],row[indexes[0]],row[8]] for row in temp])
        return Xy