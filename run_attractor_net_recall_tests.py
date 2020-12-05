###
# Will run recall tasks of MNIST on basic_hopfield, orthogonal_hebbs and popularity_ann
# MNIST will be binarized to 0 and 1
###
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from recall_task import evaluate_model_recall
from pretrained_cnn_ann import DummyCNN_ANN
from basic_hopfield import hopnet
import numpy as np

data_raw = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))

# creating a toy dataset for simple probing
mnist_subset = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
class_size = 1
for i in range(len(data_raw)):
    image, label = data_raw[i]
    if len(mnist_subset[label]) < class_size:
        mnist_subset[label].append(image)
    if all(len(k) == class_size for k in mnist_subset.values()):
        break

full_pattern_set = []
full_label_set = []
for k in mnist_subset:
    for v in mnist_subset[k]:
        full_pattern_set.append(v.reshape(-1,1).numpy())
        full_label_set.append(k)
print(full_pattern_set)
full_pattern_set = np.array(full_pattern_set)[:3]

# evaluate recall on model by storing 1 subset and then recalling
ann_model = hopnet(full_pattern_set.shape[1])
model = DummyCNN_ANN(ann_model)
num_succ, num_fail = evaluate_model_recall(model, full_pattern_set, full_label_set, full_pattern_set, full_label_set, verbose=True)
print(num_succ, ":", num_fail)


