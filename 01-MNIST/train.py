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

    Inspect the dataset: Show samples of the dataset to inspect them.

    Data loader: It is a function that we'll use to grab our data in specified
        batch sizes during training. Usually, we have a data loader for each
        dataset.

    Model Design: Construct the Neural Network. In this case a Convolutional
        Neural Network is used to recognize handwritten digits.

    Optimisers: Gradient descent algorthm or Optimiser to train the model.

    Loss Function: We need to define what type of loss we'll be using and what
        method will be using to update the gradients. We use Cross Entropy Loss
        as it is a multi-class problem.

    Training: Train the CNN tailored for digit recognition. It can be divided
        into these several steps:
        1. Get a Mini-batch of 128 training instances (images with labels);
        2. Initialise Gradients with zero values;
        3. Forward pass and take outputs;
        4. Compute loss using outputs;
        5. Backward pass;
        6. Update Gradients using the Optimiser.
        7. Repeat 1-6 to cover the entire training set;
        8. Repeat 7 for N epochs.

    Model Saving: Saving the model allows us to use it in the future to continue
        training or make inferences, without having to repeat the training
        every time.

    Performance Evaluation: Use of standard metrics to evaluate the model:
        accuracy, precision, recall, and F1 score.
        See model_evaluation.py.
"""

import torch
# dataset manipulation and image transformations
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim  # PyTorch's optimization library
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import *
from model import CNN

# device selection
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Using device: ", device)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# images transformer                                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]  # RGB: mean: (0.5, 0.5, 0.5),
                                                      # std: (0.5,0.5,0.5)
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# load our Training Data applying transformations
train_set = torchvision.datasets.MNIST('mnist',
                                       train=True,
                                       download=True,  # from PyTorch's datasets
                                       transform=transform)

# load our Test Data applying transformations
# (sometimes can be different form those applied to training set)
test_set = torchvision.datasets.MNIST('mnist',
                                      train=False,
                                      download=True,
                                      transform=transform)

# dataset inspection
print(f"Training set size: {train_set.data.shape}")
print(f"Test set size: {test_set.data.shape}")
img_show("MNIST Sample", train_set.data[0].numpy())
show_collage(train_set)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# data loader                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
train_loader = DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,  # True to prevent data sequence bias
    num_workers=0  # specifies how many CPU cores we wish to utilize (0 == main process)
)

test_loader = DataLoader(
    test_set,
    batch_size=128,
    shuffle=False,
    num_workers=0
)
# get a random batch form data loader
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(images.shape)
print(labels.shape)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Neural Network creation                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
net = CNN()
# move the model (memory and operations) to the CUDA device (or CPU/RAM)
net.to(device)
# view architecture
print(net)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Optimiser and Loss                                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# loss function
criterion = nn.CrossEntropyLoss()
# optimiser
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Training                                                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# number of iterations
epochs = 10

# log arrays
epoch_log = []
loss_log = []
accuracy_log = []

# training
for epoch in range(epochs):
    print(f'Starting Epoch: {epoch + 1}...')

    # accumulating our loss after each mini-batch in running_loss
    running_loss = 0.0

    # iterate through mini-batch
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data  # (images, labels)

        # move data to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # clear the gradients before each batch (for a fresh start)
        optimizer.zero_grad()

        # Forward -> backprop + optimize
        outputs = net(inputs)  # forward propagation
        loss = criterion(outputs, labels)  # calculate Loss
        loss.backward()  # back propagation to obtain the new gradients
        optimizer.step()  # update the gradients/weights

        # training statistics - epochs/iterations/loss/accuracy
        running_loss += loss.item()
        if i % 50 == 49:    # show our loss every 50 mini-batches
            correct = 0  # count for the correct predictions
            total = 0  # count of the number of labels iterated

            with torch.no_grad():  # no need of gradients for validation step
                # Iterate through the test_loader iterator
                for test_data in test_loader:
                    images, labels = test_data

                    # Move our data to GPU
                    images = images.to(device)
                    labels = labels.to(device)

                    # forward propagation
                    outputs = net(images)

                    # get predictions from the maximum value of the predicted output tensor
                    # dim=1: specifies the number of dimensions to reduce
                    _, predicted = torch.max(outputs, dim=1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                epoch_num = epoch + 1
                actual_loss = running_loss / 50
                print(f'Epoch: {epoch_num},'
                      f'Mini-Batches Completed: {(i+1)}, '
                      f'Loss: {actual_loss:.3f},'
                      f'Test Accuracy = {accuracy:.3f}%')
                running_loss = 0.0

    # store training stats after each epoch
    epoch_log.append(epoch_num)
    loss_log.append(actual_loss)
    accuracy_log.append(accuracy)

print('End Training!')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Model Saving                                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
PATH = './saved_models/cnn_model.pth'
torch.save(net.state_dict(), PATH)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Performance Evaluation                                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# load the model
net = CNN()
net.to(device)
net.load_state_dict(torch.load(PATH))  # load weights

# Accuracy vs Loss
fig, ax1 = plt.subplots()
plt.title("Accuracy & Loss vs Epoch")
plt.xticks(rotation=45)  # title and x-axis label rotation
ax2 = ax1.twinx()
ax1.plot(epoch_log, loss_log, 'g-')  # plot for loss_log and
ax2.plot(epoch_log, accuracy_log, 'b-')  # plot for accuracy_log
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='g')
ax2.set_ylabel('Test Accuracy', color='b')
plt.show()
