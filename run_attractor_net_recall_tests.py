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
per_class_sizes = {0:0, 1:10, 2:0, 3:0, 4:0, 5:0, 6:10, 7:0, 8:0, 9:10}
for i in range(len(data_raw)):
    image, label = data_raw[i]
    if len(mnist_subset[label]) < per_class_sizes[label]:
        mnist_subset[label].append(image)
    done = True
    for k in mnist_subset:
        if len(mnist_subset[k]) < per_class_sizes[k]:
            done=False
    if done:
        break

# given probes of a desired label, randomly choose an example of each label from the mnist dataset to store
desired_stored_labels = [1]
full_stored_set = []
full_stored_labels = []
for des in desired_stored_labels:
    desired_vectors = mnist_subset[des]
    rand_index = np.random.randint(0, len(desired_vectors))
    full_stored_set.append(desired_vectors[rand_index].reshape(-1, 1).numpy())
    full_stored_labels.append(des)

full_pattern_set = []
full_label_set = []
for k in mnist_subset:
    for v in mnist_subset[k]:
        full_pattern_set.append(v.reshape(-1,1).numpy())
        full_label_set.append(k)
full_pattern_set = np.array(full_pattern_set)

ann_model = hopnet(full_pattern_set.shape[1]) # idk what mnist dimensions are
model = DummyCNN_ANN(ann_model)
num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_pattern_set, full_label_set, verbose=True)
print(num_succ, ":", num_fail)


