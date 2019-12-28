import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
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
        # self.fc1 = nn.Linear(3552, 256)
        # self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = self.relu2(self.maxpool2(self.cnn2(out)))
        #out = out.view(out.size(0),-1)
        out = out.reshape(out.size(0), -1)
        # out = self.fc1(out)
        # out = self.fc2(out)
        # out = torch.sigmoid(out)
        return out

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_size = 64
        self.num_layers = 2
        self.lstm = nn.LSTM(3552, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
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

    sessNum = ["01","02","03","05","06","07","12","13","15","16"]
    #sessNum = ["13"]

    y_train = torch.from_numpy(np.zeros((1,1))).float()
    x_train = torch.from_numpy(np.zeros((1,150,12,1))).float()
    for sess in sessNum:
        data,label = cnnData.data_for_cnn(sess)
        x_train = torch.cat([x_train,torch.from_numpy(data).float()])
        y_train = torch.cat([y_train,torch.from_numpy(label).float()])
        print(x_train.shape)
        print(y_train.shape)

    x_train = x_train.view(-1,1,150,12)
    train = torch.utils.data.TensorDataset(x_train,y_train)

    dataset_loader = torch.utils.data.DataLoader(dataset=train,
                                                        batch_size=256,
                                                        shuffle=True)

    # training_data = data[1:,:]
    # print(training_data.shape)
    #
    # training_data = torch.tensor(training_data)
    # print(training_data)
    #
    # transformations = transforms.Compose([transforms.ToTensor()])
    #
    # custom_data =  CustomDataset(training_data)
    #
    # dataset_loader = torch.utils.data.DataLoader(dataset=custo    m_data,
    #                                                 batch_size=10,
    #                                                 shuffle=True)
    #
    model = Combine()
    model.cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(15):
        correct = 0
        startTime = time.time()
        for i, (data, label) in enumerate(dataset_loader):
            data = data.to('cuda')
            label = label.to('cuda')
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, label)
            predicted = torch.cuda.FloatTensor(outputs.data)
            array = predicted.to('cpu').numpy()
            array[array > 0.5] = 1
            array[array < 0.5] = 0
            predicted = torch.from_numpy(array).cuda()
            correct = correct + (predicted == label).sum()
            loss.backward()
            optimizer.step()
        print('Epoch : {} [({:.0f}%)] \tTime Taken: {:.4f}sec\tLoss: {:.6f}\tCorrect: {:.4f}  \tAccuracy:{:.3f}%'.format(
                        epoch, 100.*i / len(dataset_loader),time.time()-startTime , loss, float(correct),  float(correct*100) / float(256*(i+1))))
