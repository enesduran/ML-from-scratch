def f1_score(y_test,y_predicted):
    tp,fp,tn,fn=0
    for i in range(len(y_test)):
        if y_test==0 and y_predicted==0:
            tn+=1
        elif y_test==0 and y_predicted==1:
            fp+=1
        elif y_test==1 and y_predicted==1:
            tp+=1
        else:
            fn+=1
    recall=tp/fn+tp 
    precision=tp/tp+fp
    f1=2*precision*recall/(precision+recall)
    return f1

