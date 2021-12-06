import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from loader import loader
import dippykit as dip

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(8464, 1200)
        self.fc2 = nn.Linear(1200, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=-1)
        return x


net = Net()

def train():
    net = Net()

    ts = loader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(ts, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #print (inputs)
            #print (labels)
            inputs = torch.unsqueeze(inputs, dim=0)

            # zero the parameter gradients
            optimizer.zero_grad()


            # forward + backward + optimize
            outputs = net(inputs)#.unsqueeze(dim=0)

            hotvec = torch.zeros(6)
            hotvec[int(labels)] = 1

            #print(outputs)
            loss = criterion(outputs, labels)
            #loss = torch.neg(outputs).dot(hotvec)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')



    PATH = './im_class.pth'
    torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    train()