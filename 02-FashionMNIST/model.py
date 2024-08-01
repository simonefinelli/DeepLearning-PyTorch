import torch.nn as nn  # basic building block for our Network graphs
import torch.nn.functional as F  # this module contains all the functions in the torch.nn library

"""
TIP: when we use dropout and batch normalization as regularization methods,
we could respect these rules:

1. Dropout is commonly added after CONV-RELU Layers.
    - CONV -> RELU -> DROPOUT
    - Values of 0.1 to 0.3 have been found to work well
    
2. BatchNorm is best used between the Conv Layer and the activation function layer (ReLU):
    - CONV_1 -> BatchNorm -> ReLU -> Dropout -> CONV_2
    - BatchNorm's input argument is the output size of the previous layer.
"""


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        # adding BatchNorm, using 32 as the input since 32 was the output of our first Conv layer
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # adding BatchNorm, using 64 as the input since 64 was the output of our second Conv layer
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop_out = nn.Dropout(0.2)  # best suited after ReLU

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.drop_out(x)
        x = self.drop_out(F.relu(self.conv2_bn(self.conv2(x))))

        x = self.pool(x)
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

