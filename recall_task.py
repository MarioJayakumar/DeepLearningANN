#####################################################################
# This class will have utility functions for running the Recall task
# Assumes provided model follows pretrained_cnn_ann.py interface
#####################################################################
# Consider labeled images are stored in the cnn_ann model
#    A probe with label K is used
#    Recall is successful if the model converges to an attractor associated with K
#    Recall has failed if the model converges to a spurious attractor or incorrect attractor
###

from pretrained_cnn_ann import CNN_ANN

# returns 1 if recall successful
# returns 0 if recall fails
def recall_probe(model:CNN_ANN, probe, probe_label, verbose=False):
    final_act, act_label = model.predict(probe)
    if verbose:
        print("Model predicted", act_label, "- Expected:", probe_label)
    if probe_label == act_label:
        return 1
    return 0

# will first train the model on input_data and input_labels
# will then evaluate the recall of each probe in the probe set
# returns tuple of (recall successes, recall failures)
def evaluate_model_recall(model:CNN_ANN, input_data, input_labels, probe_set, probe_labels, verbose=False):
    model.learn(input_data, input_labels)
    num_succ = 0
    num_fail = 0
    for p in range(len(probe_set)):
        probe = probe_set[p]
        p_label = probe_labels[p]
        res = recall_probe(model, probe, p_label, verbose=verbose)
        if res == 1:
            num_succ += 1
        else:
            num_fail += 1
    return (num_succ, num_fail)