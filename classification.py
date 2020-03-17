import csv 
import random
import numpy as np
np.set_printoptions(precision=2)

def classifier(threshold,fileName):
    new_data=csv.writer(open("Classified_Admission.csv","w",newline=""))   
    with open(fileName) as f:
        data = list(csv.reader(f))
        for row in data:
            if float(row[8])<threshold:
               row[8]=1
            else:
               row[8]=0
            new_data.writerow(row)    
    return

def split_data(test_data_size,reg_train,class_train,reg_test,class_test):
    # initialization of writer objects 
    class_test_data=csv.writer(open(class_test,"w",newline=""))
    reg_test_data=csv.writer(open(reg_test,"w",newline=""))
    reg_train_data=csv.writer(open(reg_train,"w",newline=""))
    class_train_data=csv.writer(open(class_train,"w",newline="")) 
    reg_data=list(csv.reader(open("Admission_Predict.csv","r")))
    class_data=list(csv.reader(open("Classified_Admission.csv","r")))
    # shuffling the data and 
    random.shuffle(class_data)
    random.shuffle(reg_data)
    for i in range(501-test_data_size):
        reg_train_data.writerow(reg_data[i])
        class_train_data.writerow(class_data[i])
        pass
    for i in range(500-test_data_size,500):   
        class_test_data.writerow(class_data[i])
        reg_test_data.writerow(reg_data[i])
    return 

def PCA(fileName):
    file=open(fileName)
    file=list(csv.reader(file))
    # Serial no and the probability are ignored for PCA 
    arr=np.zeros((500,6))
    for i in range(500):
        for j in range(6):
            arr[i][j]=file[i][j+1]
    print(str(arr))          
    singular_values=np.linalg.svd(arr, full_matrices=False, compute_uv=False)
    principal_components=np.multiply(singular_values,singular_values)
    print('Lengths of the principal components are '+str(principal_components)+'\n')
    return
