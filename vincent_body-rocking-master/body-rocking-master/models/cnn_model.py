# File: cnn_model.py
# Author: Ronil Pancholia
# Date: 3/23/19
# Time: 5:52 PM

from torch import nn
from config import FREQUENCY, TIME_WINDOW

class Cnn_Model(nn.Module):
    def __init__(self):
        super(Cnn_Model, self).__init__()
        self.conv1 = nn.Conv2d(12, 96, kernel_size=(6, 1), stride=1, padding=1)
        self.conv2 = nn.Conv2d(96, 96*2, kernel_size=(4, 1), stride=1, padding=1)
        self.conv3 = nn.Conv2d(96*2, 96*4, kernel_size=(3, 1), stride=2, padding=0)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(192)
        self.bn2 = nn.BatchNorm2d(96*4)
        self.adaptivePool = nn.AdaptiveAvgPool2d((24, 1))
        self.fc1 = nn.Linear(9216, 1024)
        self.fc2 = nn.Linear(1024, 2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.view(-1, 12, FREQUENCY * TIME_WINDOW, 1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.bn1(x)
        x = self.act(self.conv3(x))
        x = self.adaptivePool(x)
        x = self.bn2(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc2(x)
        return x
