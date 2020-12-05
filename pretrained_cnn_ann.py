###############################################
# Generic harness for combining any CNN and ANN
###############################################
# Given a pretrained cnn and untrained ann model
# This class will provide two primary functions
# learn(inputs, labels)
#   Will determing the output of cnn(inputs)
#   Store those outputs in the ANN model
#   Map the stored activation
##
import numpy as np
from attractor_net import AttractorNetwork
import torch.nn as nn

# class will be used to store forward hook storage of pytorch model
class Intermediate_Capture:

    def __init__(self, module):
        self.capture = module.register_forward_hook(self.capture_function)

    def capture_function(self, module, forward, output):
        self.forward = forward
        self.output = output

    def close(self):
        self.capture.remove()

class CNN_ANN:

    # cnn: Pretrained CNN model that takes in input
    # ann: Initialized but untrained attactor net model.
    # hook: pytorch forward hook for getting output of intermediate layer
    # layer_capture: reference to a Intermediate_Capture object on cnn
    # capture_process_fn: function, that if defined, is applied to CNN intermediate output
    def __init__(self, cnn, ann:AttractorNetwork, layer_capture, capture_process_fn=None):
        self.cnn = cnn
        self.cnn.eval() # ensures that no further training occurs on cnn
        self.ann = ann
        self.layer_capture = layer_capture
        self.capture_process_fn = capture_process_fn

    def get_cnn_intermediate(self, in_data):
        self.cnn(in_data)
        captured = self.layer_capture.output.detach().numpy()
        if self.capture_process_fn is not None:
            captured = self.capture_process_fn(captured)
        return captured
    
    def learn(self, inputs, labels, verbose=False):
        inter = []
        for m in inputs:
            inter_calc = self.get_cnn_intermediate(m).reshape(self.ann.N)
            inter.append(inter_calc)
            if verbose:
                print(inter_calc)
        self.ann.learn(np.array(inter), labels)
    
    def predict(self, data_input):
        inter = self.get_cnn_intermediate(data_input).reshape(self.ann.N)
        return self.ann.simulate(inter)

# class that will be used when an ANN should be tested on its ownw
class DummyCNN_ANN(CNN_ANN):
    def __init__(self, ann:AttractorNetwork):
        self.cnn = None
        self.ann = ann

    def learn(self, inputs, labels, verbose=False):
        inter = []
        for m in inputs:
            inter_calc = m.reshape(self.ann.N)
            inter.append(inter_calc)
            if verbose:
                print(inter_calc)
        self.ann.learn(np.array(inter), labels)
    
    def predict(self, data_input):
        inter = data_input.reshape(self.ann.N)
        return self.ann.simulate(inter)

    def get_cnn_intermediate(self, in_data):
        return in_data
    