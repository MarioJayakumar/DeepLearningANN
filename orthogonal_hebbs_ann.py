################
# Orthogonal Hebbs ANN
################
# This ANN is an implementation of https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0105619
#   Specifically, section 2
# Similar to the Hopfield Network, this network uses hebbian learning to store patterns
#   One major change is that stored patterns are orthogonalized to current memories before being stored
#   Orthogonalization is done via the Gramm-Schmidt process
#   While the memories are stored orthogonalized, the recall is still done on the original memory

import numpy as np
from collections import defaultdict
from attractor_net import AttractorNetwork

# Hamming dist between vectors a, b


def hamdist(a, b): return sum(a != b)  # Hamming dist vectors a, b

# Hopfield Network Class Definition


class OrthogonalHebbsANN(AttractorNetwork):

    def __init__(self, N):              # create N-node Hopfield Net
        self.A = np.zeros(N)                 # A = activity state
        self.W = np.zeros((N, N))             # W = N x N weight matrix
        self.E = 0                        # E = current network energy
        self.label_map = defaultdict(lambda: -1, {})
        self.N = N

    # if activation contains values other than -1, 1
    # then return the sign(standardize(activation))
    def discretize_activation(self, activation):
        needs_discrete = False
        for a in activation:
            if a != float(1) or a != float(-1):
                needs_discrete = True
        if needs_discrete:
            mean = activation.mean()
            std = activation.std()
            activation = (activation-mean)/std
            activation = np.sign(activation)
        return activation

    # returns hashable encoding of the activation
    def encode_state(self, activation):
        encoding = ""
        for a in activation:
            encoding += str(a)
        return encoding

    # given input of M examples, each of length N
    # will normalize data according to gramm-schmidt process
    # returns list of M vectors, each of length N
    def orthogonalize_data(self, data):
        mu = []
        mu.append(data[0])
        for i in range(1, data.shape[0]):
            mu_sub = np.copy(data[i])
            total_sum = np.zeros(self.N)
            for j in range(i):
                ortho_ele = np.inner(data[i], mu[j]) / \
                    np.inner(mu[j], mu[j])*mu[j]
                total_sum += ortho_ele
            mu.append(mu_sub - total_sum)
        return np.array(mu)

    # simple implementation of kronecker delta
    def kronecker_delta(self, left, right):
        return int(left == right)

    # data should be M examples, where the ith example is vector of length N
    # so Data is MxN
    def learn(self, data, labels):          
        N = self.N
        M = data.shape[0]
        for i in range(M):
            self.label_map[data[i].tobytes()] = labels[i]
        # would like to turn into matrix math, but will use for loop for now
        orthogonalized_data = self.orthogonalize_data(data)
        for i in range(N-1):
            for j in range(i, N):
                w_sum = 0.0
                for k in range(M):
                    n_norm_i = orthogonalized_data[k][i]/np.linalg.norm(orthogonalized_data[k][i])
                    n_norm_j = orthogonalized_data[k][j]/np.linalg.norm(orthogonalized_data[k][j])
                    w_sum += n_norm_i*n_norm_j - self.kronecker_delta(i, j)*n_norm_i*n_norm_i
                self.W[i][j] = w_sum
                self.W[j][i] = w_sum
        
    def sgn(self, input, oldval):         # compute a = sgn(input)
        if input > 0:
            return 1
        elif input < 0:
            return -1
        else:
            return oldval

    def update(self):                       # asynchronously update A
        indices = np.random.permutation(self.N)  # determine order node updates
        for i in indices:                     # for each node i
            scalprod = np.dot(self.W[i, :], self.A)         # compute i's input
            self.A[i] = self.sgn(scalprod, self.A[i])   # assign i's act. value
        return

    def simulate(self, Ainit):
        Ainit = self.discretize_activation(Ainit)
        # Simulate Hopfield net starting in state Ainit.
        # Returns iteration number tlast and Hamming distance dist
        # of A from stored pattern Ainit when final state reached.
        # trace = 1 prints in fileid state A and energy E at each t
        t = 0                        # initialize time step t
        self.A = np.copy(Ainit)         # assign initial state A
        self.E = self.energy()       # compute energy E
        #fileid.write("self.A = {} \n".format(self.A))
        # self.showstate(fileid,t,self.E)
        Aold = np.copy(Ainit)           # A at previous t
        #fileid.write("Aold = {} \n".format(Aold))
        moretodo = True              # not known to be at equilibrium yet
        while moretodo:              # while fixed point not reached
            t += 1  # increment iteration counter
            self.update()  # update all A values once per t
            #fileid.write("self.A after updating = {} \n".format(self.A))
            #fileid.write("Aold after updating = {} \n".format(Aold))
            self.E = self.energy()  # compute energy E of state A
            #fileid.write("self.A after energy calc = {} \n".format(self.A))
            #fileid.write("Aold after energy calc = {} \n".format(Aold))
            if all(self.A == Aold):  # if at fixed point
                #fileid.write("self.A == Aold satisfied\n")
                tlast = t  # record ending iteration
                dist = hamdist(Ainit, self.A)    # distance from Ainit
                moretodo = False  # and quit
            Aold = np.copy(self.A)
            #fileid.write("Aold after updating = {} \n".format(Aold))
        #fileid.write("after while termination: self.A = {}, Aold = {} \n".format(self.A,Aold))
        # self.showstate(fileid,t,self.E)
        predict_label = self.label_map[self.A.astype(int).tobytes()]
        return (self.A, predict_label)

    def energy(self):             # Returns network's energy E
        return -0.5 * np.dot(self.A, np.dot(self.W, self.A))

    def showstate(self, fileid, t, E):  # display A at time t
        fileid.write("t: {} ".format(t))
        for i in range(self.N):
            if self.A[i] == 1:
                fileid.write("+")
            else:
                fileid.write("-")
        fileid.write("  E: {} ".format(E))
        fileid.write("\n")

    def showwts(self, fileid):    # display W
        fileid.write("\nWeights =\n")
        for i in range(self.N):
            for j in range(self.N):
                fileid.write(" {0:7.3f}".format(self.W[i, j]))
            fileid.write("\n")
