import math
import numpy as np

class Random_Forest():
    """Random forest class which consists of Decision_Tree() class"""
    # X is the and y is class label
    def __init__(self,number_of_trees=15, min_info_gain=0.05, min_number_of_division=2):
        self.number_of_trees=number_of_trees
        self.min_info_gain=min_info_gain
        self.min_number_of_division=min_number_of_division
        
        # initialization of decision trees 
        for i in range(number_of_trees):
            pass
    
    def eesplit_data():
        
        return
    
    def information_gain(y,y1,y2,min_info_gain):
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
    
    # random division of data samples and features with replacement
    def get_random_subsets(X,y,subset_number):
        # number of samples 
        n_samples = np.shape(X)[0]
        # Concatenate x and y and do a random shuffle
        Xy = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
        np.random.shuffle(Xy)
        subsets = []
        subsample_size = n_samples
        # filling the subsets
        for i in range(subset_number):
            idx = np.random.choice(range(n_samples),size=np.shape(range(subsample_size)))
            X = Xy[idx][:, :-1]
            y = Xy[idx][:, -1]
            subsets.append([X, y])
        return subsets