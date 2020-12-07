###
# Will run recall tasks of MNIST on basic_hopfield, orthogonal_hebbs and popularity_ann
# MNIST will be binarized to 0 and 1
###
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from recall_task import evaluate_model_recall
from pretrained_cnn_ann import DummyCNN_ANN, CNN_ANN, Intermediate_Capture
from basic_hopfield import hopnet
from popularity_ann import PopularityANN
from orthogonal_hebbs_ann import OrthogonalHebbsANN
from alexnet import AlexNet
import torch
import torch.nn as nn
import numpy as np

# given a list of desired labels
# will pick one of each label from full_dataset and create a subset for storing data
# returns tuple of (desired vector set, desired vector label set)
def create_probe_set(desired_labels, full_dataset, reshape=True, make_numpy=True):
    full_stored_set = []
    full_stored_labels = []
    for des in desired_labels:
        desired_vectors = full_dataset[des]
        rand_index = np.random.randint(0, len(desired_vectors))
        to_add = desired_vectors[rand_index]
        if reshape:
            to_add = to_add.reshape((-1,1))
        if make_numpy:
            to_add = to_add.numpy()
        full_stored_set.append(to_add)
        full_stored_labels.append(des)
    return (full_stored_set, full_stored_labels)

# performance series is a list of tuples, i-th element of tuple is i-th series
def draw_performance_graph(performance_series, series_labels):
    x_axis = list(range(1, len(performance_series)+1))
    full_series_set = list(map(list, zip(*performance_series)))
    for s in range(len(full_series_set)):
        series = full_series_set[s]
        lab = series_labels[s]
        plt.plot(x_axis, series, label=lab)
    plt.legend()
    plt.xlabel("Number Images Stored")
    plt.ylabel("Recall Success over Images Stored")
    plt.show()

def run_ann_recall_test_simulation():

    data_raw = MNIST('./data/mnist',
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()]))

    # creating a toy dataset for simple probing
    mnist_subset = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    per_class_sizes = {0:10, 1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10}
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
    return stored_size_vs_performance


def run_alexnet_ann_recall_test_simulation():
    transform = transforms.ToTensor()
    data_raw = MNIST(
    root='./data/mnist',
    train=True,
    download=True,
    transform=transform)

    # creating a toy dataset for simple probing
    mnist_subset = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    per_class_sizes = {0:10, 1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10}
    for i in range(len(data_raw)):
        image, label = data_raw[i]
        if len(mnist_subset[label]) < per_class_sizes[label]:
            mnist_subset[label].append(torch.reshape(image, (1,1, 28,28)))
        done = True
        for k in mnist_subset:
            if len(mnist_subset[k]) < per_class_sizes[k]:
                done=False
        if done:
            break

    # instantiate alexnet from mnist trained
    alex_cnn = AlexNet()
    alex_cnn.load_state_dict(torch.load("trained_models/alexnet.pt", map_location=torch.device("cpu")))
    alex_cnn.eval()
    alex_capture = Intermediate_Capture(alex_cnn.fc3) # for now capture final output

    # converts mnist_subset into table that is usable for model input
    full_pattern_set = []
    full_label_set = []
    for k in mnist_subset:
        for v in mnist_subset[k]:
            full_pattern_set.append(v)
            full_label_set.append(k)

    # given list of a desired labels, randomly choose an example of each label from the mnist dataset to store
    stored_size_vs_performance = [] # list will store tuples of (hopfield perf, popularity perf, ortho perf)
    for desired_label_size in range(10):
        desired_labels = list(range(desired_label_size+1))
        full_stored_set, full_stored_labels = create_probe_set(desired_labels, mnist_subset, reshape=False, make_numpy=False)
        print("Num Stored: ", len(desired_labels))

        num_nodes = 10 # TODO: should extract from Intermediate Capture size

        # evaluate hopnet performance
        ann_model = hopnet(num_nodes) 
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=True)
        print("Hopfield:", num_succ, ":", num_fail)
        hopfield_perf = int(num_succ)

        # evaluate popularity ANN performance
        # hyperparams: set c = N-1, with randomly generated connectivity matrix
        ann_model = PopularityANN(N=num_nodes, c=num_nodes-1)
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=True)
        print("PopularityANN:", num_succ, ":", num_fail)
        popularity_perf = int(num_succ)

        # evaluate orthogonal hebbs ANN performance
        ann_model = OrthogonalHebbsANN(N=num_nodes)
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=True)
        print("OrthogonalHebbsANN:", num_succ, ":", num_fail)
        ortho_perf = int(num_succ)

        stored_size_vs_performance.append((hopfield_perf, popularity_perf, ortho_perf))

    # write performance to file
    fh = open("data/graph_sources/alexnet_recall_task_trial1.txt", "w")
    for perf in stored_size_vs_performance:
        fh.write(str(perf[0]) + "," + str(perf[1]) + "," + str(perf[2]) + "\n")
    fh.close()
    return stored_size_vs_performance

# data must be stored in CSV
# each row represents tuple of (hopfield recall success, popularity recall success, orthogonal recall success)
def parse_recall_task_perf(src_file):
    fh = open(src_file, "r")
    stored_size_vs_performance = []
    num_stored = 1
    for line in fh:
        data = line.split(",")
        if len(data)>0:
            hopfield_perf = float(int(data[0]) / num_stored)
            popularity_perf = float(int(data[1]) / num_stored)
            orthogonal_perf = float(int(data[2]) / num_stored)
            stored_size_vs_performance.append((hopfield_perf, popularity_perf, orthogonal_perf))
        num_stored += 1
    fh.close()
    return stored_size_vs_performance

draw_performance_graph(parse_recall_task_perf("data/graph_sources/mnist_recall_task_anns.txt"), series_labels=["Hopfield", "Popularity ANN", "Orthogonal ANN"])

#run_alexnet_ann_recall_test_simulation()
#draw_performance_graph(parse_recall_task_perf("data/graph_sources/alexnet_recall_task_trial1.txt"), series_labels=["Hopfield", "Popularity ANN", "Orthogonal ANN"])
