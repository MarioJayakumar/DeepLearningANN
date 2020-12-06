###
# Will run recall tasks of MNIST on basic_hopfield, orthogonal_hebbs and popularity_ann
# MNIST will be binarized to 0 and 1
###
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from recall_task import evaluate_model_recall
from pretrained_cnn_ann import DummyCNN_ANN
from basic_hopfield import hopnet
from popularity_ann import PopularityANN
from orthogonal_hebbs_ann import OrthogonalHebbsANN
import numpy as np

# given a list of desired labels
# will pick one of each label from full_dataset and create a subset for storing data
# returns tuple of (desired vector set, desired vector label set)
def create_probe_set(desired_labels, full_dataset):
    full_stored_set = []
    full_stored_labels = []
    for des in desired_labels:
        desired_vectors = full_dataset[des]
        rand_index = np.random.randint(0, len(desired_vectors))
        full_stored_set.append(desired_vectors[rand_index].reshape(-1, 1).numpy())
        full_stored_labels.append(des)
    return (full_stored_set, full_stored_labels)

# performance series is a list of tuples, i-th element of tuple is i-th series
def draw_performance_graph(performance_series):
    x_axis = list(range(len(performance_series)))
    full_series_set = map(list, zip(*performance_series))
    for series in full_series_set:
        plt.plot(x_axis, series)
    plt.show()

data_raw = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))

# creating a toy dataset for simple probing
mnist_subset = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
per_class_sizes = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1}
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

# converts mnist_subset into table that is usable for model input
full_pattern_set = []
full_label_set = []
for k in mnist_subset:
    for v in mnist_subset[k]:
        full_pattern_set.append(v.reshape(-1,1).numpy())
        full_label_set.append(k)
full_pattern_set = np.array(full_pattern_set)

# given list of a desired labels, randomly choose an example of each label from the mnist dataset to store
stored_size_vs_performance = [] # list will store tuples of (hopfield perf, popularity perf, ortho perf)
for desired_label_size in range(10):
    desired_labels = list(range(desired_label_size+1))
    full_stored_set, full_stored_labels = create_probe_set(desired_labels, mnist_subset)
    print("Num Stored: ", len(desired_labels))

    num_nodes = full_pattern_set.shape[1] 

    # evaluate hopnet performance
    ann_model = hopnet(num_nodes) 
    model = DummyCNN_ANN(ann_model)
    num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=True)
    print("Hopfield:", num_succ, ":", num_fail)
    hopfield_perf = float(num_succ/(desired_label_size+1))

    # evaluate popularity ANN performance
    # hyperparams: set c = N-1, with randomly generated connectivity matrix
    ann_model = PopularityANN(N=num_nodes, c=num_nodes-1)
    model = DummyCNN_ANN(ann_model)
    num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=True)
    print("PopularityANN:", num_succ, ":", num_fail)
    popularity_perf = float(num_succ/(desired_label_size+1))

    # evaluate orthogonal hebbs ANN performance
    ann_model = OrthogonalHebbsANN(N=num_nodes)
    model = DummyCNN_ANN(ann_model)
    num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=True)
    print("OrthogonalHebbsANN:", num_succ, ":", num_fail)
    ortho_perf = float(num_succ/(desired_label_size+1))

    stored_size_vs_performance.append((hopfield_perf, popularity_perf, ortho_perf))


draw_performance_graph(stored_size_vs_performance)
