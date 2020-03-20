import math
import numpy as np
from DecisionTree import DecisionTree

class Random_Forest(DecisionTree):
    """Random forest class which consists of Decision_Tree() class"""
    # X is the and y is class label
    def __init__(self,number_of_trees=15, min_info_gain=0.05, min_number_of_division=2):
        self.number_of_trees=number_of_trees
        self.min_info_gain=min_info_gain
        self.min_number_of_division=min_number_of_division
        # list of decision trees
        forest=[]
    
        # initialization of decision trees 
        for i in range(number_of_trees):
            
            
            forest.append(train())
            pass
    
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
    
    # training all decision trees the incoming data 
    def train():
        tree=DecisionTree()
        # filling the subsets
        for i in range(subset_number):
        tree.__init__()
        return tree
   
    # make prediction by voting
    def predict(self,test_data,number_of_trees):
        prediction=np.zeros(test_data.shape[0])
        for i in range(len(prediction)):
            for j in range(number_of_trees):
                pass
        return 
    
    # random division of data samples and features with replacement
    def get_random_subsets(X,y,subsample_size):
        # names of features for visulaizing tree
        feature_names=['GRE','TOEFL','University Rating','SOP','LOR','CGPA','Research']
        feature_number_for_each_tree=math.ceil(math.sqrt(len(feature_names)))
        # number of samples 
        n_samples = np.shape(X)[0]
        # Concatenate x and y and do a random shuffle
        Xy = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
        np.random.shuffle(Xy)
        subsets = []
        subsample_size = n_samples
        
            idx = np.random.choice(range(n_samples),size=np.shape(range(subsample_size)))
            X = Xy[idx][:, :-1]
            y = Xy[idx][:, -1]
            feature_1
            feature_2
            subsets.append([X, y,feature_1,feature_2])
        return subsets