import numpy as np
import lenet
import torch
import torch.nn as nn
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Load trained LeNet model
# Compare MNIST input for normal output and ANN output
data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True)
data_test_loader = DataLoader(data_test, batch_size=1)

lenet_model = lenet.LeNet5()
lenet_model.load_state_dict(torch.load("trained_models/lenet5_1.pt"))
lenet_model.eval()

for test_images, test_labels in data_test_loader:
    test_images_pic = test_images.reshape((32, 32)).numpy()
    test_labels = test_labels.numpy()
    predicted_label = torch.argmax(torch.exp(lenet_model.forward(test_images)))
    predicted_intermediate = lenet_model.ANN_forward(test_images)
    print(predicted_label)
    print(predicted_intermediate)
    plt.imshow(test_images_pic)
    plt.show()