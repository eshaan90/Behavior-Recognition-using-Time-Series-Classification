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

    sessNum = ["01","02","03","05","06","07","12","13","15","16"]
    
    y_train = torch.from_numpy(np.zeros((1,1))).float()
    x_train = torch.from_numpy(np.zeros((1,150,12,1))).float()
    for sess in sessNum:
        data,label = cnnData.data_for_cnn(sess)
        x_train = torch.cat([x_train,torch.from_numpy(data).float()])
        y_train = torch.cat([y_train,torch.from_numpy(label).float()])

    x_train = x_train.view(-1,1,150,12)
    train = torch.utils.data.TensorDataset(x_train,y_train)

    validation_split = 0.1
    random_seed= 42
    dataset_size = len(train)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                        batch_size=64,
                                                        sampler = train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset=train,
                                                        batch_size=64,
                                                        sampler = valid_sampler)

    print("Length of Training Data : ",len(train_loader))
    print("Length of Validation Data : ",len(valid_loader))

    train_loss_data = []
    train_accuracy_data = []
    test_loss_data = []
    test_accuracy_data = []
    model = Combine()
    model.cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(30):
        correct = 0
        startTime = time.time()
        total_loss = torch.FloatTensor([0])
        for i, (data, label) in enumerate(train_loader):
            data = data.to('cuda')
            label = label.to('cuda')
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, label)
            total_loss += loss
            predicted = torch.cuda.FloatTensor(outputs.data)
            array = predicted.to('cpu').numpy()
            array[array > 0.7] = 1
            array[array < 0.7] = 0
            predicted = torch.from_numpy(array).cuda()
            correct = correct + (predicted == label).sum()
            loss.backward()
            optimizer.step()
        model.eval()
        val_correct = 0
        total_val_loss = torch.FloatTensor([0])
        for j, (val_data, val_label) in enumerate(valid_loader):
            val_data = val_data.to('cuda')
            val_label = val_label.to('cuda')
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, val_label)
            total_val_loss+=val_loss.data
            val_predicted = torch.cuda.FloatTensor(val_outputs.data)
            array = val_predicted.to('cpu').numpy()
            array[array > 0.7] = 1
            array[array < 0.7] = 0
            val_predicted = torch.from_numpy(array).cuda()
            val_correct = val_correct + (val_predicted == val_label).sum()
        model.train()
        test_loss_data.append((total_val_loss.data)/j)
        test_accuracy_data.append(float(val_correct*100)/float(64*(j+1)))
        train_loss_data.append((total_loss.data)/i)
        train_accuracy_data.append(float(correct*100)/float(64*(i+1)))
        torch.save(model.state_dict(), os.path.join(os.getcwd()+"/Model/", 'epoch-{}.pt'.format(epoch+1)))
        print('Epoch : {} \tTime Taken: {:.4f}sec\tTraining Loss: {:.6f} \tTraining Accuracy:{:.3f} \tValidation Loss: {:.6f} \tValidation Accuracy:{:.3f}%'.format(
                        epoch+1, time.time()-startTime , loss, float(correct*100) / float(64*(i+1)),val_loss, float(val_correct*100) / float(64*(j+1))))

    plt.figure(1)
    plt.plot(train_loss_data, label='Training loss')
    plt.plot(test_loss_data, label='Validation loss')
    plt.ylabel("Loss")
    plt.ylim((0, 3))
    plt.xlabel("Epochs")
    plt.legend(frameon=False)
    plt.show()

    plt.figure(2)
    plt.plot(train_accuracy_data, label='Training Accuracy')
    plt.plot(test_accuracy_data, label='Validation Accuracy')
    plt.ylabel("Accuracy")
    plt.ylim((50, 100))
    plt.xlabel("Epochs")
    plt.legend(frameon=False)
    plt.show()
