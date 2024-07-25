"""
This script evaluates the performance of the pre-trained Convolutional Neural
Network (CNN) on the MNIST test.

The steps involved include:
    1. Loading the pre-trained model's weights from a saved file.
    2. Setting the model to evaluation mode to disable specific layers that
        behave differently during training and inference.
    3. Preparing the test dataset and data loader with appropriate
        transformations.
    4. Computing the accuracy of the model on the test dataset by comparing the
        predicted labels with the true labels.
    5. Insight into performances using Precision, Recall and F1 score.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report

from torch.utils.data import DataLoader
from model import CNN
from utils import *

# device selection
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device: {device}")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Model Loading                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
PATH = './saved_models/cnn_model.pth'
net = CNN()
net.to(device)
net.load_state_dict(torch.load(PATH))  # load weights


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Performance Evaluation                                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
net.eval()  # turn off specific layers/parts (Dropouts Layers, BatchNorm
            # Layers etc.) of the model that behave differently during training
            # and inference (evaluating step)

# replicate test-set dataloader
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_set = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

# Test Accuracy
correct = 0
total = 0
with torch.no_grad():  # use no_grad to save memory (gradients are unuseful during test)
    for (images, labels) in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f'Accuracy of the network on the test set: {accuracy:.3}%')

# Confusion Matrix
pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
label_list = torch.zeros(0, dtype=torch.long, device='cpu')
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)

        # Append batch prediction results
        pred_list = torch.cat([pred_list, preds.view(-1).cpu()])
        label_list = torch.cat([label_list, labels.view(-1).cpu()])

show_confusion_matrix(label_list, pred_list,
                      list(range(0, 10)), list(range(0, 10)))

# Classification Report
print(classification_report(label_list.numpy(), pred_list.numpy()))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# View Misclassification                                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
print(f"Misclassified images on Test Set:")
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        predictions = torch.argmax(outputs, dim=1)

        # For test data in each batch we identify when predictions did not match the label
        # then we print out the actual ground truth
        for i in range(data[0].shape[0]):
            pred = predictions[i].item()
            label = labels[i]
            if label != pred:
                print(f'Actual Label: {pred}, Predicted Label: {label}')
