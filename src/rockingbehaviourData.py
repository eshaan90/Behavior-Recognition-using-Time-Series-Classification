import os
import numpy as np

def data_for_cnn(sessNum):
    dataPath = os.getcwd()+"/Training Data B/Session" + sessNum + "/"

    detection = []
    detec = open(dataPath+"detection.txt",'r')
    for line in detec:
        detection.append(line)
    detection = np.array(detection).astype(np.float)

    armData = []
    file = open(dataPath+"armIMU.txt",'r')
    for line in file:
        armData.append(line.strip().split("  "))
    data1 = np.array(armData).astype(np.float)

    wristData = []
    file = open(dataPath+"wristIMU.txt",'r')
    for line in file:
            wristData.append(line.strip().split("  "))
    data2 = np.array(wristData).astype(np.float)

    combined_data = np.zeros((data1.shape[0],data1.shape[1] +  data2.shape[1]))
    combined_data[:,0:6] = data1
    combined_data[:,6:] = data2

    label = []
    stride = 25
    count = 0
    len = 150
    detCount = 74
    while(count<combined_data.shape[0]-len):
        label.append(detection[detCount])
        detCount+=stride
        count+=stride
    label.append(detection[combined_data.shape[0]-75])
    labelLength = np.array(label).shape[0]
    labels = np.zeros((labelLength,1))
    labels[:,0] = np.array(label)

    cnn_data = np.zeros((labels.shape[0],150,12,1))
    count = 0
    while(count<combined_data.shape[0]-len):
        cnn_data[int(count/stride),:,:,0] = combined_data[count:count+len,:]
        count+=stride

    cnn_data[-1,:,:,0] = combined_data[combined_data.shape[0]-len:combined_data.shape[0],:]

    return (cnn_data,labels)

def data_for_test(sessNum):
    dataPath = os.getcwd()+"/Val Data 2/Session" + sessNum + "/"
    armData = []
    file = open(dataPath+"armIMU.txt",'r')
    for line in file:
        armData.append(line.strip().split("  "))
    data1 = np.array(armData).astype(np.float)
    print("ARMIMU data : ",data1.shape)
    wristData = []
    file = open(dataPath+"wristIMU.txt",'r')
    for line in file:
        wristData.append(line.strip().split("  "))
    data2 = np.array(wristData).astype(np.float)
    print("WRISTIMU data : ",data2.shape)
    combined_data = np.zeros((data1.shape[0]+149,data1.shape[1] +  data2.shape[1]))
    print(combined_data.shape)
    combined_data[149:,0:6] = data1
    combined_data[149:,6:] = data2

    label = []
    stride = 1
    count = 0
    len = 150

    cnn_data =np.zeros((data1.shape[0],150,12,1))
    count = 0
    while(count<combined_data.shape[0]-len):
        cnn_data[int(count/stride),:,:,0] = combined_data[count:count+len,:]
        count+=stride
    print(count)
    print(cnn_data.shape)
    return cnn_data
