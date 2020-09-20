import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 5)
        self.batchNorm1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 48, 3)
        self.batchNorm2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 24, 3)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 26 * 26, 189)
        self.fc2 = nn.Linear(189, 126)
        self.fc3 = nn.Linear(126, 84)
        self.fc4 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.batchNorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchNorm2(self.conv2(x))))
        x = self.pool(F.relu(self.batchNorm3(self.conv3(x))))
        x = x.view(x.size()[0], 24 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
