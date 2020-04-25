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
    print('The recall is {0:.2f} \n'.format(recall))
    print('The precision is {0:.2f} \n'.format(precision))
    return f1