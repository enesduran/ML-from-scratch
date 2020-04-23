import numpy as np
import math

class Node:
        # if not initialized it will be counted as leaf node
        def __int__(self,feature_index,threshold=None,right=None,left=None,value=None):
            # threshold for division
            self.threshold=threshold
            # index of criteria feature for division of branches.
            self.feature_index=feature_index
            # the values higher than threshold will be put right node
            self.right=right
            # left subtreee
            self.left=left
            # the value of the node if it is leaf
            if right is None and left is None:
                self.value=value
            else:
                self.value=None
            return 
        
class DecisionTree:
    
    def __init__(self,Xy,feature_1,feature_2,min_info_gain,min_number_of_division):
        # declare a root node to build tree
        feature_names=['GRE','TOEFL','University Rating','SOP','LOR','CGPA','Research']
        self.features=[0,0]
        # feature_1 & 2 are the indexes of the features
        self.features[0]=feature_1
        self.features[1]=feature_2
        self.min_info_gain=min_info_gain
        self.min_number_of_division=min_number_of_division
        self.Xy=Xy
        self.root=self.train(Xy)
            
    def train(self,X_y):
        """Building tree by constructing nodes. Calls split_by_feature method to split the data """
        r,_=X_y.shape
        if r==1:
            return Node()
        else:
            for feature_number in range(len(self.features)):
                best_split_gain=0    
                unique_vals=np.unique(X_y[:,feature_number])
                #print(unique_vals)
                for threshold in unique_vals:
                    # Splitted data according to threshold
                    Xr,Xl=self.split_by_feature(X_y,feature_number,threshold)
                    if not (len(X_y)==0 or len(Xr)==0 or len(Xl)==0):
                        # Now it is time to look the information gain to justify split 
                        temp=X_y[:,2]
                        temp_r=np.array(Xr[row[2]] for row in Xr)
                        temp_l=np.array(Xl[row[2]] for row in Xl)
                        print(temp_r)
                        print(temp_l)
                        split_gain=self.information_gain(temp,temp_r,temp_l)
                        # Check whether it is the best split 
                        if split_gain>best_split_gain:
                            best_split_gain=split_gain
                            idx=feature_number
                            best_threshold=threshold
        return Node(idx,best_threshold)
      
    # recursive method    
    def split_by_feature(self,data,feature_number,threshold):
        """Divide the data into two according to the threshold on feature number"""
        X_left=[]
        X_right=[]
        for row in data:
            if row[feature_number]>=threshold:
                X_left.append(row)
            else:
                X_right.append(row)
                print(X_left)
        return X_right,X_left
   
    def predict(self,x,node=None):
        """Makes the prediction of single data point"""
        if node is None:
            node=self.root
        else:
            # This means the node is leaf node
            if (node.left is None and node.right is None):
                return node.value
            # continue searching
            else:
                # left branch
                if x[node.feature_index]<node.threshold:
                    branch=node.left    
                # right branch    
                else:
                    branch=node.right
        return self.predict(x,branch)
    
    # calculate info gain 
    def information_gain(self,y,y1,y2):
        # y1 is empty
        prob=len(y1)/len(y)
        info_gain=self.entropy(y)-self.entropy(y1)*prob+self.entropy(y2)*(1-prob)
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
     
    def predictionArray(self,X_test):
        """returns the predictions in an array"""
        y_predict=[]
        # Filling the array of prediction
        for row in X_test:
            y_predict.append(self.predict(row))
        return y_predict