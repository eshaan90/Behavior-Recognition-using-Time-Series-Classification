import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import time

import rockingbehaviourData as cnnData

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, stride=3),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, 64, kernel_size=3, stride=3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=5, stride=1 , padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        return out

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_size = 64
        self.num_layers = 1
        self.lstm = nn.LSTM(2304, 64, 1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.cnn = CNNModel()
        self.rnn = RNN()

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0),1,-1)
        out = self.rnn(out)
        out = torch.sigmoid(out)
        return out

if __name__ == "__main__":

    model = Combine()
    model.load_state_dict(torch.load('epoch-17.pt'))
    model.cuda()
    model.eval()
    print(model)
    sessNum = ["08","09","11","17"]
    #sessNum = ["11"]
    for sess in sessNum:
        x_test = torch.from_numpy(np.zeros((1,150,12,1))).float()
        data= cnnData.data_for_test(sess)
        x_test = torch.cat([x_test,torch.from_numpy(data).float()])

        x_test = x_test.view(-1,1,150,12)
        x_test = x_test[1:]
        print("Data Set Size : " , x_test.shape)

        predicted = np.zeros(x_test.shape[0])
        startTime = time.time()
        for i , data_test in enumerate(x_test):
            data = torch.from_numpy(np.zeros((1,1,150,12))).float()
            data[0,0,:,:] = data_test
            #data = x_test[i,0,:,:]
            data = data.cuda()
            output = model(data)
            pred = output.data
            array = pred.to('cpu').numpy()
            if(array<0.7):
                predicted[i] = 0
            else:
                predicted[i] = 1


        f = open("predicted{}.txt".format(sess) , 'w')
        for pred in predicted:
            f.write(str(pred))
            f.write("\n")
        f.close()
        print("Time Taken for Predicting Session {} is {:.3f}".format(sess , time.time()-startTime))
