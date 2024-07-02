import torch.nn as nn  # basic building block for our Network graphs
import torch.nn.functional as F  # this module contains all the functions in the torch.nn library


class CNN(nn.Module):
    def __init__(self):
        # super is a subclass of the nn.Module and inherits all its methods
        super(CNN, self).__init__()

        # define the entire network
        # first CNN Layer using 32 Fitlers of 3x3 size, with stride of 1 & padding of 0
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 0)
        # second CNN Layer using 64 Fitlers of 3x3 size, with stride of 1 & padding of 0
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Max Pool Layer 2 x 2 kernel of stride 2
        self.pool = nn.MaxPool2d(2, 2)
        # first Fully Connected Layer, takes the output of our Max Pool
        # which is 12 x 12 x 64 and connects it to a set of 128 nodes
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        # second Fully Connected Layer, connects the 128 nodes to 10 output nodes (our classes)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # link together forward propagation sequence (NN structure):
        # Conv1 - Relu - Conv2 - Relu - Max Pool - Flatten - FC1 - FC2
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
