import numpy as np
np.set_printoptions(precision=2)

# model evaluation method for SVM and Random Forest  
def f1_score(y_test,y_predicted):
    tp,fp,tn,fn=0,0,0,0
    for i in range(len(y_test)):
        if (float(y_test[i])==0.0 and y_predicted[i]==0):
            tn+=1
        elif (float(y_test[i])==0.0 and y_predicted[i]==1):
            fp+=1
        elif (float(y_test[i])==1.0 and y_predicted[i]==1):
            tp+=1
        else:
            fn+=1
    recall=tp/(fn+tp) 
    precision=tp/(tp+fp)
    f1=2*precision*recall/(precision+recall)
    print('Recall: {0:.2f}'.format(recall),' Precision: {0:.2f}'.format(precision))
    return f1

from RandomForest import Random_Forest

def average_f1_score(model="RandomForest"):
    """ Average f1 score is measured by training 5 random models
        and averaging their f1 scores.""" 
    print("\nRandom Forest Initialization")    
    f1=0
    for i in range(5):
        print(i,"'th run")
        forest=Random_Forest()
        f1=f1+forest.returnf1()
    fNew=f1/5    
    print("Average f1 score of the RandomForest after 5 run: ",fNew)    
    return fNew