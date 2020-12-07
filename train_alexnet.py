import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from alexnet import AlexNet
 
# Define whether to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Hyper parameter settings
EPOCH = 10 # Number of times to traverse the data set
BATCH_SIZE = 64 # Batch size (batch_size)
LR = 0.01 # learning rate
 
 # Define the data preprocessing method
transform = transforms.ToTensor()
 
 # Define training data set
trainset = tv.datasets.MNIST(
    root='./data/mnist',
    train=True,
    download=True,
    transform=transform)
 
 # Define training batch data
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
 
 # Define test data set
testset = tv.datasets.MNIST(
    root='./data/mnist',
    train=False,
    download=True,
    transform=transform)
 
 # Define test batch data
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
 
 # Define loss function loss function and optimization method (using SGD)
net = AlexNet().to(device)
criterion = nn.CrossEntropyLoss() # Cross entropy loss function, usually used for multi-classification problems
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
 
 # Train and save model parameters
def train():
 
    for epoch in range(EPOCH):
        sum_loss = 0.0
                 # Data read
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
 
                         # Gradient clear
            optimizer.zero_grad()
 
            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
                         # Print the average loss every 100 batches trained
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
                 # Test the accuracy after every epoch
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # The category with the highest score
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('The recognition accuracy of the %d epoch is: %d%%'% (epoch + 1, (100 * correct / total)))
                # Save model parameters
        torch.save(net.state_dict(), './trained_models/alexnet.pt')
 
if __name__ == "__main__":
    train()
 
