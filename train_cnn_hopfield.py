import numpy as np
import lenet
import torch
import torch.nn as nn
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from basic_hopfield import hopnet
from pretrained_cnn_ann import CNN_ANN, Intermediate_Capture


# Load trained LeNet model
# Compare MNIST input for normal output and ANN output
data_raw = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))

unique_train_images = []
unique_train_labels = []
for i in range(len(data_raw)):
    image, label = data_raw[i]
    if label not in unique_train_labels:
        unique_train_images.append(image.reshape((1,1,32,32)))
        unique_train_labels.append(label)
    if len(unique_train_labels) > 2:
        break

lenet_model = lenet.LeNet5()
lenet_model.load_state_dict(torch.load("trained_models/lenet5_1.pt"))
lenet_model.eval()
lenet_capture = Intermediate_Capture(lenet_model.f5)
hopfield_net = hopnet(10)
test_b = CNN_ANN(lenet_model, hopfield_net, lenet_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))

test_b.learn(unique_train_images, unique_train_labels, verbose=True)

for train_image in unique_train_images:
    print(test_b.predict(train_image))
