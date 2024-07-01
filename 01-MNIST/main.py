"""
Project Goal: Training a CNN on the MNIST Dataset

In this project, we develop a Convolutional Neural Network (CNN) to recognize
handwritten digits using the MNIST dataset. The MNIST dataset is a well-known
benchmark in the field of machine learning and consists of 60,000 training
images and 10,000 testing images of handwritten digits, labeled from 0 to 9.
The primary objective of this project is to design and train an effective
CNN model to achieve high accuracy in digit classification. Additionally,
we aim to analyze the model's performance and extract meaningful insights from
the training process and results.

Key Objectives:
    Imports: import the libraries.

    Transformer definition: Transformers are needed to ensure the instance
        (image) has the right format for input into our model.
        Typically, we transform the image to a PyTorch tensor and normalizing
        the image values between -1 and +1.
        Normalization is done like this: image = (image - mean) / std

    Dataset definition: Normally we retrieve three subsets from the data:
        - Training set: Data that is used during training
        - Test/Validation set: Data that is used to evaluate the model performance
        TIP: MNIST doesn't have a separate test set. Therefore, we use the test
        set for both validation and test.

    Inspect the dataset:

    Data loader:

    Model Design and Training: Construct and train a CNN tailored for digit
        recognition using PyTorch library.

    Performance Evaluation: Use of the Confusion Matrix to evaluate accuracy,
        precision, recall, and F1 score.
"""

import torch
# dataset manipulation and image transformations
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim  # PyTorch's optimization library
import torch.nn as nn  # basic building block for our Network graphs

from utils import *

# check for GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# images transformer
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
    # for RGB: (0.5, 0.5, 0.5), (0.5,0.5,0.5)
)

# load our Training Data applying transformations
train_set = torchvision.datasets.MNIST('mnist',
                                       train=True,
                                       download=True,  # from PyTorch's datasets
                                       transform=transform)

# load our Test Data applying transformations (sometimes can be different form those applied to training set)
test_set = torchvision.datasets.MNIST('mnist',
                                      train=False,
                                      download=True,
                                      transform=transform)

# dataset inspection
print(f"Training set size: {train_set.data.shape}")
print(f"Test set size: {test_set.data.shape}")
img_show("MNIST Sample", train_set.data[0].numpy())
show_collage(train_set)




