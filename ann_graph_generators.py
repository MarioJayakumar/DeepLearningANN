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

def draw_mnist_img(img):
    img = img.reshape((32,32))
    plt.imshow(img, cmap='gray')
    plt.show()

def draw_mnist_comparison(original, hopfield, popularity, orthogonal):
    fig, axes = plt.subplots(ncols=10, nrows=4, constrained_layout=True)
    index = 0
    for orig in original:
        axes.ravel()[index].imshow(orig.reshape((32, 32)), cmap='gray')
        axes.ravel()[index].axis('off')
        index += 1
    for orig in hopfield:
        axes.ravel()[index].imshow(orig.reshape((32, 32)), cmap='gray')
        axes.ravel()[index].axis('off')
        index += 1
    for orig in popularity:
        axes.ravel()[index].imshow(orig.reshape((32, 32)), cmap='gray')
        axes.ravel()[index].axis('off')
        index += 1
    for orig in orthogonal:
        axes.ravel()[index].imshow(orig.reshape((32, 32)), cmap='gray')
        axes.ravel()[index].axis('off')
        index += 1
    plt.show()

# given a list of desired labels
# will pick one of each label from full_dataset and create a subset for storing data
# returns tuple of (desired vector set, desired vector label set)
def create_probe_set(desired_labels, full_dataset, N):
    full_stored_set = []
    full_stored_labels = []
    for des in desired_labels:
        desired_vectors = full_dataset[des]
        rand_index = np.random.randint(0, len(desired_vectors))
        full_stored_set.append(desired_vectors[rand_index].reshape(N).numpy())
        full_stored_labels.append(des)
    return (full_stored_set, full_stored_labels)

def draw_mnist_recall(data_raw):
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
    desired_labels = list(range(10))
    num_nodes = full_pattern_set.shape[1] 
    full_stored_set, full_stored_labels = create_probe_set(desired_labels, mnist_subset, N=num_nodes)
    full_stored_set = np.array(full_stored_set)


    # evaluate hopnet performance
    ann_model = hopnet(num_nodes) 
    ann_model.learn(full_stored_set, full_stored_labels)
    hopfield_produced_images = []
    for probe in full_stored_set:
        hopfield_produced_images.append(ann_model.simulate(probe)[0])
    print("Done Hopfield")
    original_images = []
    for probe in full_stored_set:
        original_images.append(ann_model.threshold(probe, theta=0.5))

    # evaluate popularity ANN performance
    # hyperparams: set c = N-1, with randomly generated connectivity matrix
    ann_model = PopularityANN(N=num_nodes, c=num_nodes-1)
    ann_model.learn(full_stored_set, full_stored_labels)
    print("Done Popularity Learning")
    popularity_produced_images = []
    for probe in full_stored_set:
        popularity_produced_images.append(ann_model.simulate(probe)[0])
    print("Done Popularity")
    # evaluate orthogonal hebbs ANN performance
    ann_model = OrthogonalHebbsANN(N=num_nodes)
    ann_model.learn(full_stored_set, full_stored_labels)
    orthogonalANN_produced_images = []
    for probe in full_stored_set:
        orthogonalANN_produced_images.append(ann_model.simulate(probe)[0])
    print("Done Orthogonal")
    draw_mnist_comparison(original_images, hopfield_produced_images, popularity_produced_images, orthogonalANN_produced_images)



data_raw = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
draw_mnist_recall(data_raw)