"""
Project Goal: Training a CNN on the FashionMNIST Dataset

In this project, we develop a Convolutional Neural Network (CNN) to recognize
various clothing items using the FashionMNIST dataset. The FashionMNIST dataset
is a well-known benchmark in the field of machine learning and consists of
60,000 training images and 10,000 testing images of fashion items, labeled from
0 to 9, with each label corresponding to a specific type of clothing.

The primary objective of this project is to design and train an effective CNN
model to achieve high accuracy in fashion item classification. The model will
use the following Regularisation methods:
- L2 Regularisation
- Data Augmentation
- Dropout
- BatchNorm

Additionally, we aim to analyze the model's performance and extract meaningful
insights from the training process and results.

Key Objectives:
    Transformer definition: Transformers are needed to ensure the instance
        (image) has the right format for input into our model.
        Typically, we transform the image to a PyTorch tensor and normalizing
        the image values between -1 and +1.
        Normalization is done like this: image = (image - mean) / std.
        In this case we will apply Data Augmentation to improve the model
        generalization.

    Dataset definition: Normally we retrieve three subsets from the data:
        - Training set: Data that is used during training.
        - Test/Validation set: Data that is used to evaluate the model performance.
        TIP: FashionMNIST doesn't have a separate test set. Therefore, we use
        the test set for both validation and test.

    Inspect the dataset: Show samples of the dataset to inspect them.

    Data loader: It is a function that we'll use to grab our data in specified
        batch sizes during training. Usually, we have a data loader for each
        dataset.

    Model Design: Construct the Neural Network. In this case a Convolutional
        Neural Network is used to recognize fashion clothing.

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
import PIL

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
# Images transformer                                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
data_transforms = {
    'train': transforms.Compose([
        # note: these transforms are executed in the order they are called here
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, interpolation=PIL.Image.BILINEAR),
        transforms.Grayscale(num_output_channels=1),  # some of these transforms return a color image,
                                                      # hence why we need to convert the image back to grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # RGB: mean: (0.5, 0.5, 0.5),
                                                        # std: (0.5,0.5,0.5)
    ])
}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# load our Train Data and specify what transform to use when loading
train_set = torchvision.datasets.FashionMNIST(
    root='./fashion_mnist', train=True, download=True, transform=data_transforms['train'])

# load our Test Data and specify what transform to use when loading
test_set = torchvision.datasets.FashionMNIST(
    root='./fashion_mnist', train=False, download=True, transform=data_transforms['val'])

# test data augmentation
test_augmentations(train_set.data[0].numpy(), data_transforms, 6)

# dataset inspection
print(f"Training set size: {train_set.data.shape}")
print(f"Test set size: {test_set.data.shape}")
img_show("FashionMNIST Sample", train_set.data[0].numpy())
show_collage(train_set)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Data Loader                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,  # True to prevent data sequence bias
    num_workers=2  # specifies how many CPU cores we wish to utilize (0 == main process)
)

test_loader = DataLoader(
    test_set,
    batch_size=32,
    shuffle=False,
    num_workers=2
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
"""
TIP: L2 regularisation on the parameters/weights of the model is directly 
included in most optimizers (like optim.SGD). It can be controlled with the 
weight_decay parameter:
    - weight_decay (L2 penalty): good values range from 0.1 to 0.0001 (default: 0)

NOTE: L1 regularization is not included by default in the optimizers, but could 
be added by including an extra for the weights of the model.
"""
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Training                                                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# number of iterations
epochs = 15

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
fig.savefig('./saved_models/accuracy_vs_loss.png')
