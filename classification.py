import csv 
import random

def classifier(threshold):
    new_data=csv.writer(open("Classified_Admission.csv","w",newline=""))   
    with open('Admission_Predict.csv') as f:
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
