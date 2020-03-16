import numpy as np 
import logging

def kNN(k,reg_train_data,reg_test_data,test_data_size):
    predictions=np.array(test_data_size)
    # calculating the closest k data points in training set
    for x in range(test_data_size):
        # the distances & indexes first k closest point
        closestK=np.array(k)
        indexes=np.array(k)
        for r in range(sum(1 for row in reg_train_data)):
            # distance is l2 norm
            distance=0
            for y in range(7):
                distance+=(reg_train_data[r][y]-reg_test_data[x][y])**2
            if distance<closestK[0]:  
                indexes[0]=r
                closestK[0]=distance
                sorting(closestK,indexes)
                pass
        # now we have calculated the closest k training points for a certain test point    
        predict=0 
        for t in range(k):    
            predict=predict+float(reg_train_data[indexes[k]][8])
        predictions[x]=predict/k
    errorCalculation(k,predictions,reg_test_data)
    logging.info("Predictions are",predictions)    
    return 

def sorting(closestK,indexes):
    for i in closestK:
        for j in range(len(closestK)-1):
            if closestK(j-1)>closestK(j):
                indexes[j-1],indexes[j]=indexes[j], indexes[j-1] 
                closestK[j-1],closestK[j]=closestK[j],closestK[j-1]
    return

def errorCalculation(predictions,reg_test_data):
    error=0
    length=len(reg_test_data)
    for i in range(length):
        error+=abs(predictions[i]-reg_test_data[i][8])/reg_test_data[i][8]
    error=error/length
    logging.info("The average prediction error percentage= ",str(error))
    return error