# File: cnn_paper.py
# Author: Ronil Pancholia
# Date: 3/24/19
# Time: 12:34 AM

from torch import nn
from config import FREQUENCY, TIME_WINDOW

class Cnn_Model3(nn.Module):
    def __init__(self):
        super(Cnn_Model3, self).__init__()
        self.conv1 = nn.Conv2d(12, 96, kernel_size=(9, 1), stride=3, padding=0)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=(7, 1), stride=3, padding=0)
        self.conv3 = nn.Conv2d(192, 300, kernel_size=(3, 1), stride=3, padding=0)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(1200, 500)
        self.fc2 = nn.Linear(500, 2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.view(-1, 12, FREQUENCY * TIME_WINDOW, 1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.fc1(x.view(x.size(0), -1))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
