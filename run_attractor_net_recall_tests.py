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
from lenet import LeNet5
import torch
import torch.nn as nn
import numpy as np

# given a list of desired labels
# will pick one of each label from full_dataset and create a subset for storing data
# returns tuple of (desired vector set, desired vector label set)
def create_storage_set(desired_labels, full_dataset, reshape=True, make_numpy=True):
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



def draw_shared_performance_graph(performance_series1, performance_series2, performance_series3, series_labels):
    x_axis = list(range(1, len(performance_series1)+1))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    full_series_set = list(map(list, zip(*performance_series1)))
    handles = []
    for s in range(len(full_series_set)):
        series = full_series_set[s]
        lab = series_labels[s]
        ax1.plot(x_axis, series, label=lab)
    full_series_set = list(map(list, zip(*performance_series2)))
    for s in range(len(full_series_set)):
        series = full_series_set[s]
        lab = series_labels[s]
        ax2.plot(x_axis, series, label=lab)
    full_series_set = list(map(list, zip(*performance_series3)))
    for s in range(len(full_series_set)):
        series = full_series_set[s]
        lab = series_labels[s]
        ax3.plot(x_axis, series, label=lab)
    ax1.set_ylabel("Recall Success over Images Stored")
    ax2.set_xlabel("Number Images Stored")
    plt.legend()
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
        full_stored_set, full_stored_labels = create_storage_set(desired_labels, mnist_subset)
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

# provide capture to specify output layer to utilize
def run_alexnet_ann_recall_simulation(alex_cnn, alex_capture, output_name, num_nodes):
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
        full_stored_set, full_stored_labels = create_storage_set(desired_labels, mnist_subset, reshape=False, make_numpy=False)
        print("Num Stored: ", len(desired_labels))

        # evaluate hopnet performance
        ann_model = hopnet(num_nodes) 
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=False)
        print("Hopfield:", num_succ, ":", num_fail)
        hopfield_perf = int(num_succ)

        # evaluate popularity ANN performance
        # hyperparams: set c = N-1, with randomly generated connectivity matrix
        ann_model = PopularityANN(N=num_nodes, c=num_nodes-1)
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=False)
        print("PopularityANN:", num_succ, ":", num_fail)
        popularity_perf = int(num_succ)

        # evaluate orthogonal hebbs ANN performance
        ann_model = OrthogonalHebbsANN(N=num_nodes)
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=False)
        print("OrthogonalHebbsANN:", num_succ, ":", num_fail)
        ortho_perf = int(num_succ)

        stored_size_vs_performance.append((hopfield_perf, popularity_perf, ortho_perf))

    # write performance to file
    fh = open("data/graph_sources/" + output_name, "w")
    for perf in stored_size_vs_performance:
        fh.write(str(perf[0]) + "," + str(perf[1]) + "," + str(perf[2]) + "\n")
    fh.close()
    return stored_size_vs_performance

def run_alexnet_ann_recall_test_simulation_trial1():
    # instantiate alexnet from mnist trained
    alex_cnn = AlexNet()
    alex_cnn.load_state_dict(torch.load("trained_models/alexnet.pt", map_location=torch.device("cpu")))
    alex_cnn.eval()
    alex_capture = Intermediate_Capture(alex_cnn.fc3) # for now capture final output
    return run_alexnet_ann_recall_simulation(alex_cnn=alex_cnn, alex_capture=alex_capture, output_name="alexnet_recall_task_trial1.txt", num_nodes=10)

def run_alexnet_ann_recall_test_simulation_trial2():
    # instantiate alexnet from mnist trained
    alex_cnn = AlexNet()
    alex_cnn.load_state_dict(torch.load("trained_models/alexnet.pt", map_location=torch.device("cpu")))
    alex_cnn.eval()
    alex_capture = Intermediate_Capture(alex_cnn.fc2) # for now capture final output
    return run_alexnet_ann_recall_simulation(alex_cnn=alex_cnn, alex_capture=alex_capture, output_name="alexnet_recall_task_trial2.txt", num_nodes=512)

def run_alexnet_ann_recall_test_simulation_trial3():
    # instantiate alexnet from mnist trained
    alex_cnn = AlexNet()
    alex_cnn.load_state_dict(torch.load("trained_models/alexnet.pt", map_location=torch.device("cpu")))
    alex_cnn.eval()
    alex_capture = Intermediate_Capture(alex_cnn.fc1) # for now capture final output
    return run_alexnet_ann_recall_simulation(alex_cnn=alex_cnn, alex_capture=alex_capture, output_name="alexnet_recall_task_trial3.txt", num_nodes=1024)

# capture output layers 3, 4, 5
def run_alexnet_ann_recall_test_simulation_trial4():
    num_nodes = 10
    alex_cnn1 = AlexNet()
    alex_cnn1.load_state_dict(torch.load("trained_models/alexnet.pt", map_location=torch.device("cpu")))
    alex_cnn1.eval()
    alex_capture1 = Intermediate_Capture(alex_cnn1.layer3) # for now capture final output
    output_name = "alexnet_recall_task_trial4.txt"

    alex_cnn2 = AlexNet()
    alex_cnn2.load_state_dict(torch.load("trained_models/alexnet.pt", map_location=torch.device("cpu")))
    alex_cnn2.eval()
    alex_capture2 = Intermediate_Capture(alex_cnn2.layer4) # for now capture final output

    alex_cnn3 = AlexNet()
    alex_cnn3.load_state_dict(torch.load("trained_models/alexnet.pt", map_location=torch.device("cpu")))
    alex_cnn3.eval()
    alex_capture3 = Intermediate_Capture(alex_cnn3.layer5) # for now capture final output

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
        full_stored_set, full_stored_labels = create_storage_set(desired_labels, mnist_subset, reshape=False, make_numpy=False)
        print("Num Stored: ", len(desired_labels))

        # evaluate hopnet performance
        ann_model = hopnet(6272) 
        model = CNN_ANN(alex_cnn1, ann_model, alex_capture1, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=False)
        print("Alexnet Layer3:", num_succ, ":", num_fail)
        layer3_perf = int(num_succ)

        ann_model = hopnet(12544) 
        model = CNN_ANN(alex_cnn2, ann_model, alex_capture2, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=False)
        print("Alexnet Layer3:", num_succ, ":", num_fail)
        layer4_perf = int(num_succ)

        ann_model = hopnet(2304) 
        model = CNN_ANN(alex_cnn3, ann_model, alex_capture3, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=False)
        print("Alexnet Layer3:", num_succ, ":", num_fail)
        layer5_perf = int(num_succ)

        stored_size_vs_performance.append((layer3_perf, layer4_perf, layer5_perf))

    # write performance to file
    fh = open("data/graph_sources/" + output_name, "w")
    for perf in stored_size_vs_performance:
        fh.write(str(perf[0]) + "," + str(perf[1]) + "," + str(perf[2]) + "\n")
    fh.close()
    return stored_size_vs_performance

def run_alexnet_ann_recall_test_simulation_trial5():
    output_name="alexnet_recall_task_trial5.txt"
    num_nodes=1024
    full_connection_mat = np.ones(shape=(num_nodes,num_nodes)) - np.eye(num_nodes)
    alex_cnn = AlexNet()
    alex_cnn.load_state_dict(torch.load("trained_models/alexnet.pt", map_location=torch.device("cpu")))
    alex_cnn.eval()
    alex_capture = Intermediate_Capture(alex_cnn.fc1) # for now capture final output

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

        # need to generate probe set each time
        # when desired label size is k:
        # probe set is 10 instances each of labels 0 to k-1
        desired_labels = list(range(desired_label_size+1))
        sub_probe_set = []
        sub_probe_labels = []
        for des in desired_labels:
            # add 10 instances of des
            for inst in mnist_subset[des]:
                sub_probe_set.append(inst)
                sub_probe_labels.append(des)
        full_stored_set, full_stored_labels = create_storage_set(desired_labels, mnist_subset, reshape=False, make_numpy=False)
        print("Num Stored: ", len(desired_labels))

        # evaluate hopnet performance
        ann_model = hopnet(num_nodes) 
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, sub_probe_set, sub_probe_labels, verbose=False)
        print("Hopfield:", num_succ, ":", num_fail)
        hopfield_perf = int(num_succ)

        # evaluate popularity ANN performance
        # hyperparams: set c = N-1, with randomly generated connectivity matrix
        ann_model = PopularityANN(N=num_nodes, c=num_nodes-1, connectivity_matrix=full_connection_mat)
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, sub_probe_set, sub_probe_labels, verbose=False)
        print("PopularityANN:", num_succ, ":", num_fail)
        popularity_perf = int(num_succ)

        # evaluate orthogonal hebbs ANN performance
        ann_model = OrthogonalHebbsANN(N=num_nodes)
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, sub_probe_set, sub_probe_labels, verbose=False)
        print("OrthogonalHebbsANN:", num_succ, ":", num_fail)
        ortho_perf = int(num_succ)

        stored_size_vs_performance.append((hopfield_perf, popularity_perf, ortho_perf))

    # write performance to file
    fh = open("data/graph_sources/" + output_name, "w")
    for perf in stored_size_vs_performance:
        fh.write(str(perf[0]) + "," + str(perf[1]) + "," + str(perf[2]) + "\n")
    fh.close()
    return stored_size_vs_performance

def run_alexnet_ann_recall_test_simulation_trial6():
    output_name="alexnet_recall_task_trial6.txt"
    num_nodes=512
    full_connection_mat = np.ones(shape=(num_nodes,num_nodes)) - np.eye(num_nodes)
    alex_cnn = AlexNet()
    alex_cnn.load_state_dict(torch.load("trained_models/alexnet.pt", map_location=torch.device("cpu")))
    alex_cnn.eval()
    alex_capture = Intermediate_Capture(alex_cnn.fc2) # for now capture final output

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

        # need to generate probe set each time
        # when desired label size is k:
        # probe set is 10 instances each of labels 0 to k-1
        desired_labels = list(range(desired_label_size+1))
        sub_probe_set = []
        sub_probe_labels = []
        for des in desired_labels:
            # add 10 instances of des
            for inst in mnist_subset[des]:
                sub_probe_set.append(inst)
                sub_probe_labels.append(des)
        full_stored_set, full_stored_labels = create_storage_set(desired_labels, mnist_subset, reshape=False, make_numpy=False)
        print("Num Stored: ", len(desired_labels))

        # evaluate hopnet performance
        ann_model = hopnet(num_nodes) 
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, sub_probe_set, sub_probe_labels, verbose=False)
        print("Hopfield:", num_succ, ":", num_fail)
        hopfield_perf = int(num_succ)

        # evaluate popularity ANN performance
        # hyperparams: set c = N-1, with randomly generated connectivity matrix
        ann_model = PopularityANN(N=num_nodes, c=num_nodes-1, connectivity_matrix=full_connection_mat)
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, sub_probe_set, sub_probe_labels, verbose=False)
        print("PopularityANN:", num_succ, ":", num_fail)
        popularity_perf = int(num_succ)

        # evaluate orthogonal hebbs ANN performance
        ann_model = OrthogonalHebbsANN(N=num_nodes)
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, sub_probe_set, sub_probe_labels, verbose=False)
        print("OrthogonalHebbsANN:", num_succ, ":", num_fail)
        ortho_perf = int(num_succ)

        stored_size_vs_performance.append((hopfield_perf, popularity_perf, ortho_perf))

    # write performance to file
    fh = open("data/graph_sources/" + output_name, "w")
    for perf in stored_size_vs_performance:
        fh.write(str(perf[0]) + "," + str(perf[1]) + "," + str(perf[2]) + "\n")
    fh.close()
    return stored_size_vs_performance

def run_alexnet_ann_recall_test_simulation_trial7():
    output_name="alexnet_recall_task_trial7.txt"
    num_nodes=10
    full_connection_mat = np.ones(shape=(num_nodes,num_nodes)) - np.eye(num_nodes)
    alex_cnn = AlexNet()
    alex_cnn.load_state_dict(torch.load("trained_models/alexnet.pt", map_location=torch.device("cpu")))
    alex_cnn.eval()
    alex_capture = Intermediate_Capture(alex_cnn.fc3) # for now capture final output

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

        # need to generate probe set each time
        # when desired label size is k:
        # probe set is 10 instances each of labels 0 to k-1
        desired_labels = list(range(desired_label_size+1))
        sub_probe_set = []
        sub_probe_labels = []
        for des in desired_labels:
            # add 10 instances of des
            for inst in mnist_subset[des]:
                sub_probe_set.append(inst)
                sub_probe_labels.append(des)
        full_stored_set, full_stored_labels = create_storage_set(desired_labels, mnist_subset, reshape=False, make_numpy=False)
        print("Num Stored: ", len(desired_labels))

        # evaluate hopnet performance
        ann_model = hopnet(num_nodes) 
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, sub_probe_set, sub_probe_labels, verbose=False)
        print("Hopfield:", num_succ, ":", num_fail)
        hopfield_perf = int(num_succ)

        # evaluate popularity ANN performance
        # hyperparams: set c = N-1, with randomly generated connectivity matrix
        ann_model = PopularityANN(N=num_nodes, c=num_nodes-1, connectivity_matrix=full_connection_mat)
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, sub_probe_set, sub_probe_labels, verbose=False)
        print("PopularityANN:", num_succ, ":", num_fail)
        popularity_perf = int(num_succ)

        # evaluate orthogonal hebbs ANN performance
        ann_model = OrthogonalHebbsANN(N=num_nodes)
        model = CNN_ANN(alex_cnn, ann_model, alex_capture, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, sub_probe_set, sub_probe_labels, verbose=False)
        print("OrthogonalHebbsANN:", num_succ, ":", num_fail)
        ortho_perf = int(num_succ)

        stored_size_vs_performance.append((hopfield_perf, popularity_perf, ortho_perf))

    # write performance to file
    fh = open("data/graph_sources/" + output_name, "w")
    for perf in stored_size_vs_performance:
        fh.write(str(perf[0]) + "," + str(perf[1]) + "," + str(perf[2]) + "\n")
    fh.close()
    return stored_size_vs_performance

# capture output of layers C3, S4 and C5
def run_lenet_ann_recall_test_simulation_trial1():
    output_name = "lenet_recall_task_trial1.txt"

    lenet_cnn2 = LeNet5()
    lenet_cnn2.load_state_dict(torch.load("trained_models/lenet5_1.pt", map_location=torch.device("cpu")))
    lenet_cnn2.eval()
    lenet_capture2 = Intermediate_Capture(lenet_cnn2.c2_2) 

    lenet_cnn3 = LeNet5()
    lenet_cnn3.load_state_dict(torch.load("trained_models/lenet5_1.pt", map_location=torch.device("cpu")))
    lenet_cnn3.eval()
    lenet_capture3 = Intermediate_Capture(lenet_cnn3.c3) 

    lenet_cnn4 = LeNet5()
    lenet_cnn4.load_state_dict(torch.load("trained_models/lenet5_1.pt", map_location=torch.device("cpu")))
    lenet_cnn4.eval()
    lenet_capture4 = Intermediate_Capture(lenet_cnn4.f4) 

    transform = transforms.ToTensor()
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
            mnist_subset[label].append(torch.reshape(image, (1,1, 32,32)))
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
            full_pattern_set.append(v)
            full_label_set.append(k)

    # given list of a desired labels, randomly choose an example of each label from the mnist dataset to store
    stored_size_vs_performance = [] # list will store tuples of (hopfield perf, popularity perf, ortho perf)
    for desired_label_size in range(10):
        desired_labels = list(range(desired_label_size+1))
        full_stored_set, full_stored_labels = create_storage_set(desired_labels, mnist_subset, reshape=False, make_numpy=False)
        print("Num Stored: ", len(desired_labels))

        # evaluate hopnet performance
        ann_model = hopnet(400) 
        model = CNN_ANN(lenet_cnn2, ann_model, lenet_capture2, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=False)
        print("lenet Layer4:", num_succ, ":", num_fail)
        layer3_perf = int(num_succ)

        ann_model = hopnet(120) 
        model = CNN_ANN(lenet_cnn3, ann_model, lenet_capture3, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=False)
        print("lenet Layer5:", num_succ, ":", num_fail)
        layer4_perf = int(num_succ)

        ann_model = hopnet(84) 
        model = CNN_ANN(lenet_cnn4, ann_model, lenet_capture4, capture_process_fn=lambda x: np.sign(np.exp(x)-np.exp(x).mean()))
        num_succ, num_fail = evaluate_model_recall(model, full_stored_set, full_stored_labels, full_stored_set, full_stored_labels, verbose=False)
        print("lenet Layer6:", num_succ, ":", num_fail)
        layer5_perf = int(num_succ)

        stored_size_vs_performance.append((layer3_perf, layer4_perf, layer5_perf))

    # write performance to file
    fh = open("data/graph_sources/" + output_name, "w")
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

#draw_performance_graph(parse_recall_task_perf("data/graph_sources/mnist_recall_task_anns.txt"), series_labels=["Hopfield", "Popularity ANN", "Orthogonal ANN"])

# run_alexnet_ann_recall_test_simulation_trial2()
# run_alexnet_ann_recall_test_simulation_trial3()

p1 = parse_recall_task_perf("data/graph_sources/alexnet_recall_task_trial5.txt")
p2 = parse_recall_task_perf("data/graph_sources/alexnet_recall_task_trial6.txt")
p3 = parse_recall_task_perf("data/graph_sources/alexnet_recall_task_trial7.txt")
draw_shared_performance_graph(p1, p2, p3, series_labels=["Hopfield", "Popularity ANN", "Orthogonal ANN"])
###
# Description of Trials
###
# mnist_recall_task_anns is just storing raw MNIST images into each ANN, and viewing the recall
# alexnet_recall_trial1 is the fc3 output of alexnet combined with each ANN
# alexnet_recall_trial2 is the fc2 output of alexnet combined with each ANN
# alexnet_recall_trial3 is the fc1 output of alexnet combined with each ANN
# alexnet_recall_trial4 is alexnet layers 3,4,5 combined with Hopfield
# alexnet_recall_trial5 is trial3, except 9 additional images of each label is probed
# alexnet_recall_trial6 is trial2, except 9 additional images of each label is probed
# alexnet_recall_trial7 is trial1, except 9 additional images of each label is probed
# lenet_recall_trial1 is lenet layers C2, C3 and F4 over Hopfield

#run_alexnet_ann_recall_test_simulation()
#draw_performance_graph(parse_recall_task_perf("data/graph_sources/alexnet_recall_task_trial1.txt"), series_labels=["Hopfield", "Popularity ANN", "Orthogonal ANN"])


# print("Start Trial 6")
# run_alexnet_ann_recall_test_simulation_trial6()
# print("Start Trial 7")
# run_alexnet_ann_recall_test_simulation_trial7()
# print("Start Trial 5")
# run_alexnet_ann_recall_test_simulation_trial5()
