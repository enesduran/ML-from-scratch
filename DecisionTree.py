import numpy as np
import math

class Node:
        # if not initialized it will be counted as leaf node
        def __init__(self,feature_index=None,depth=None,threshold=None,right=None,left=None,leaf_value=None):
            # index of criteria feature for division of branches.
            self.feature_index=feature_index
            # depth of the decision node
            self.depth=depth
            # threshold for division
            self.threshold=threshold
            # the values higher than threshold will be put right node
            self.right=right
            # left subtreee
            self.left=left
            # The node is not a leaf node. Hence feature index is important
            self.leaf_value=self.leaf_value=leaf_value
            # the value of the node if it is leaf
            if right is None and left is None:
                self.leaf_value=leaf_value
        
class DecisionTree:
    
    def __init__(self,Xy,feature_1,feature_2,min_info_gain,max_depth=6,min_sample=5):
        # declare a root node to build tree
        feature_names=['GRE','TOEFL','University Rating','SOP','LOR','CGPA','Research']
        self.features=[0,0]
        # feature_1 & 2 are the indexes of the features
        self.features[0]=feature_1
        self.features[1]=feature_2
        self.feature_name_1=feature_names[feature_1]
        self.feature_name_2=feature_names[feature_2]
        self.min_info_gain=min_info_gain
        self.Xy=Xy
        self.max_depth=max_depth
        self.min_sample=min_sample
        self.root=self.train(Xy,0)
                  
    def train(self,X_y,cur_depth):
        """Building tree by constructing nodes. Calls split_by_feature method to split the data """
        if len(X_y)<=self.min_sample:
            # A leaf node 
            value=self.assign_label(X_y)
            return Node(leaf_value=value)
        else:
            # best split gain will be calculated with loops over features and thresholds
            best_split_gain=0
            
            # requirements for division
            if cur_depth<self.max_depth:
                print("current depth in training is ",cur_depth)
                right_data=0
                left_data=0    
                for feature_number in range(len(self.features)):
                    unique_vals=list(np.unique(np.array(X_y)[:,feature_number]))
                    #print(unique_vals)
                    for threshold in unique_vals:
                        # Splitted data according to threshold
                        Xr,Xl=self.split_by_feature(X_y,feature_number,threshold)
                        if not (X_y is None or Xr is None or Xl is None):
                            if (len(Xr)>=self.min_sample and len(Xl)>=self.min_sample):
                                # Now it is time to look the information gain to justify split 
                                temp=list([row[2] for row in X_y])
                                temp_r=list([row[2] for row in Xr])
                                temp_l=list([row[2] for row in Xl])
                                split_gain=self.information_gain(temp,temp_r,temp_l)
                                # Check whether it is the best split 
                                if split_gain>best_split_gain:
                                    # dataset update
                                    right_data,left_data=Xr,Xl
                                    best_split_gain=split_gain
                                    idx=self.features[feature_number]
                                    best_threshold=threshold
                                    print("BEST CASE: X_y {} Xr {} Xl {}".format(len(X_y),len(right_data),len(left_data)))
                            
            if best_split_gain>=self.min_info_gain:
                # recursive approach to extend 
                print("Best possible division Xr {} Xl {} best gain {}".format(len(right_data),len(left_data),best_split_gain))
                cur_depth+=1
                right_branch=self.train(right_data,cur_depth)
                left_branch=self.train(left_data,cur_depth)
                new_node=Node(idx,cur_depth,best_threshold,right_branch,left_branch)
                #print("new node threshold",best_threshold)
                return new_node
                
            # this means the node is leaf node
            leaf_label=self.assign_label(X_y)             
            return Node(leaf_value=leaf_label)
     
    def assign_label(self,data):
        """assign most repetitive label as the label of the leaf node"""
        y=[row[2] for row in data]
        label=0
        if y.count("1.0")>y.count("0.0"):
            label=1
        return label
    
    # recursive method    
    def split_by_feature(self,data,feature_number,threshold):
        """Divide the data into two according to the threshold on feature number"""
        X_left=[]
        X_right=[]
        for row in data:
            if row[feature_number]<=threshold:
                X_left.append(row)
            else:
                X_right.append(row)
        if (len(X_left)==0 or len(X_right)==0):
            [X_right,X_left]=[None,None]
        return X_right,X_left
   
    def predict(self,x,node=None):
        """Makes the prediction of single data point"""
        # starting tree by assigning node
        if node is None:
            node=self.root
            #print("sürekli root")
            
        # Check if it is a leaf node
        if not(node.leaf_value is None):
            #print("leaf")
            return node.leaf_value
        
        if (node.right is None or node.left is None):
            print("exception")
             
        thresh=node.threshold
        feature_i=node.feature_index
        # assign next branch
        if x[feature_i]<=thresh:
            #print("sola kaydı")
            branch=node.left   
        else:
            branch=node.right
            #print("sağa kaydı")
        return self.predict(x,branch)
    
    # calculate info gain 
    def information_gain(self,y,y1,y2):
        prob=len(y1)/len(y)
        info_gain=-self.entropy(y)+self.entropy(y1)*prob+self.entropy(y2)*(1-prob)
        #print("Info gain is ",info_gain )
        return info_gain
    
    # entropy calculation of a node
    def entropy(self,y):
        # there will always be 2 classes at max
        entropy=0
        # determining number of classes
        if not (y.count(0)>0) and (y.count(1)>0):
            class_number=1
            return entropy
        else:
            class_number=2 
            for i in range(class_number):
                #print("length of y is ",len(y))
                probability=len(y[y == i])/len(y)
                entropy-= probability*math.log(probability,2)
            return entropy
     
    def predictionArray(self,X_test):
        """returns the predictions in an array"""
        y_predict=[]
        # Filling the array of prediction
        for row in X_test:
            y_predict.append(self.predict(row))
        #print("y predict",y_predict)
        return y_predict