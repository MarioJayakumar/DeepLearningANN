###
# Generic Attractor Network
###
import numpy as np
from collections import defaultdict

class AttractorNetwork:

    def __init__(self, N):
        self.N = N
        self.A = np.zeros(N)
        self.label_map = defaultdict(lambda:-1, {})

    # generic function for learning a dataset
    # data must have M examples, with each example of dimension N
    # labels is list of M associated mapping for each memory
    # does not have to return anything
    def learn(self, data, labels):
        pass
    
    # generic function for iterating to an attractor state
    #   from an input activation state
    # Return: 
    #   tuple of final activation state and associated label
    def simulate(self, activation):
        pass 