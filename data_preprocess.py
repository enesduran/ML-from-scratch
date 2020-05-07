import csv 
import random
import numpy as np
np.set_printoptions(precision=2)
X=np.zeros((500,9))


def classifier(threshold,fileName,normalizedFile):
    class_new_data=csv.writer(open("Classified_Admission.csv","w",newline=""))
    reg_new_data=csv.writer(open(normalizedFile,"w",newline=""))
    old_data=list(csv.reader(open(fileName,"r")))
    data=normalize(old_data)
    for row in data:
        reg_new_data.writerow(np.around(row,2))
        if float(row[8])<threshold:
            row[8]=(0)
        else:
            row[8]=(+1)
        class_new_data.writerow(np.around(row,2))  
    return

def split_data(test_data_size,reg_train,class_train,reg_test,class_test):
    # initialization of writer objects 
    class_test_data=csv.writer(open(class_test,"w",newline=""))
    reg_test_data=csv.writer(open(reg_test,"w",newline=""))
    reg_train_data=csv.writer(open(reg_train,"w",newline=""))
    class_train_data=csv.writer(open(class_train,"w",newline="")) 
    reg_data=list(csv.reader(open("Regression_Admission.csv","r")))
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

# making mean of the dataset equal to zero
def normalize(old_data):
    m=len(old_data)
    n=len(old_data[0])
    print(m,n)
    means=np.array([0,0,0,0,0,0,0,0,0])
    # normalization over samples
    #.astype(np.float)
    for j in range(n):
        print(old_data[:][j])
        arr=[rows[j] for rows in old_data]
       
        means[j]=np.mean( np.array(arr).astype(np.float))
        print(means)
        # since the last 2 features ranges between (0,1) no need to normalize 
        for i in range(m):
            X[i][j]=float(old_data[i][j])
            # making mean equal to zero
            if j<8 and j>0:
                X[i][j]=X[i][j]-means[j]
    print("means are", means)           
    return X     

# principal component analysis
def PCA(fileName='Regression_Admission.csv'):
    # the dataset is already normalized. We directly compute its SVD       
    singular_values=np.linalg.svd(X, full_matrices=False, compute_uv=False)
    principal_components=np.multiply(singular_values,singular_values)
    print('Lengths of the principal components are '+str(principal_components)+'\n')
    return

# extracting the input vector(predictors) from the data set 
def getInputSamples(data,input_data_file):
    #data = list(csv.reader(open(data_file,'r')))
    input_data = csv.writer(open(input_data_file,"w",newline=""))
    for sample in data:
        x =[]
        for i in range(1,len(sample)-1):
            x.append(sample[i])
        input_data.writerow(x)
    return 

# extracting the output vector(labels) from the data set 
def getOutputValues(data,output_data_file):
    #data = list(csv.reader(open(data_file,'r')))
    output_data = csv.writer(open(output_data_file,"w",newline=""))
    for samples in data:
        x =[]
        x.append(samples[len(samples)-1])
        output_data.writerow(x)
    return 

# given a data set calculates both the input and output vectors for classification 
def calculateInputOutput(class_data):
    class_input="Input_Train_Data.csv"
    class_output="Output_Train_Data.csv"
    getInputSamples(class_data,class_input)
    getOutputValues(class_data,class_output)
    a = list(csv.reader(open(class_input,'r')))
    b = list(csv.reader(open(class_output,'r')))
    X=np.array(a,dtype=float)
    y=np.array(b,dtype=float)
    return [X,y] 

# converts the 1 and 0 labled data to 1 and -1 labeled data for SVM    
def convertSVM(data_list):
    new_data_list = data_list[:]
    for row in new_data_list:
        #i = new_data_list.index(row)
        if row[len(row)-1] == "0.0":
            row[len(row)-1] = "-1.0"
            #new_data_list[i] = row
    return new_data_list