import csv
import numpy as numpy
numpy.set_printoptions(precision=2) 
                    
def kNN(k,reg_train,reg_test,test_data_size):
    # changing csv files writer to reader
    reg_train_data=csv.reader(open(reg_train,'r'))
    reg_test_data=csv.reader(open(reg_test,'r'))
    reg_test_data=list(reg_test_data)
    reg_train_data=list(reg_train_data)
    # prediction array
    predictions=[0]*test_data_size
    # calculating the closest k data points in training set
    for x in range(test_data_size):
        # the distances & indexes first k closest point
        closestK=[0]*k
        indexes=[0]*k
        for r in range(sum(1 for row in reg_train_data)):
            # distance is l2 norm
            distance=0
            for y in range(1,7):
                distance+=(float(reg_train_data[r][y])-float(reg_test_data[x][y]))**2
            if r<k:
                closestK[r]=distance
                indexes[r]=r
            else: 
                if distance<closestK[0]:  
                    indexes[0]=r
                    closestK[0]=distance
                    sorting(closestK,indexes)
                    pass
        # now we have calculated the closest k training points for a certain test point    
        predict=0 
        for t in range(k):    
            predict=predict+float(reg_train_data[indexes[k-1]][8])
        predictions[x]=predict/k
    errorCalculation(predictions,reg_test_data)
    print('Predictions of kNN are')
    print(numpy.array(predictions))   
    return 

def sorting(closestK,indexes):
    for i in closestK:
        for j in range(len(closestK)-1):
            if closestK[j-1]>closestK[j]:
                indexes[j-1],indexes[j]=indexes[j], indexes[j-1] 
                closestK[j-1],closestK[j]=closestK[j],closestK[j-1]
    return

def errorCalculation(predictions,reg_test_data):
    error=0
    length=len(reg_test_data)
    for i in range(length):
        error+=abs(predictions[i]-float(reg_test_data[i][8]))/float(reg_test_data[i][8])
    error=100*error/length
    print('Average prediction error percentage of the kNN= {0:.2f} \n'.format(error))
    return error