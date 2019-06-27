# File: mlp.py
# Author: Ronil Pancholia
# Date: 3/20/19
# Time: 5:40 PM

from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(12, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 250, 12, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
