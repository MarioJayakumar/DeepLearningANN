################
# Popularity ANN
################
# This ANN is an implementation of https://www.tandfonline.com/doi/pdf/10.2976/1.2793335
# Popularity ANN addresses a few problems with the Hopfield network
#   Sparse connections are allowed, with the paper outlining how attractor stability increases with sparseness
#   Correlating memories are stabler in memory compared to Hopfield networks, due to a different update rule
#   The update rule of this network utilizes a new concept called neuron popularity

from attractor_net import AttractorNetwork
import numpy as np

# randomly generates a sparse connection matrix of dim NxN
# c represents the max number of connections per node
# if connected is True, then disconnected graph is not allowed
def generate_sparse_connection_matrix(N, c, connected=True):
    sparse = []
    connections = []
    for i in range(N):
        temp = []
        for j in range(N):
            temp.append(0)
        sparse.append(temp)
        connections.append(0)
    # only initialize lower left triangle, then make symmetric
    for i in range(N-1):
        # diagonal is always 0
        # so output size should be N - 1 - i
        des_size = N - i - 1
        p_connection = np.random.randint(2, size=des_size)
        if connected:
            while (int(np.sum(p_connection)) == 0):
                p_connection = np.random.randint(2, size=des_size)
        # make sure a dst node does not end up with too many connections
        for j in range(des_size):
            if connections[i+1+j] >= c:
                p_connection[j] = 0
        max_con_available = max(c - connections[i], 0)
        to_delete = int(np.sum(p_connection)) - max_con_available
        j = 0
        k = 0
        while j < to_delete: # this should probably be changed to random ordering, rather than sequential
            if p_connection[k] == 1: 
                p_connection[k] = 0
                j += 1
            k += 1
        # ready to copy connections to sparse matrix
        for j in range(i+1, N):
            sparse[i][j] = p_connection[j-i-1]
            sparse[j][i] = sparse[i][j] # make symmetric
            connections[j] += p_connection[j-i-1]
    return np.array(sparse)

class PopularityANN(AttractorNetwork):

    # N:integer is number of nodes
    # c:int(<N) is upperbound on mean connections per neuron
    #   will generate a sparse connection matrix
    # if connectivity_matrix(NxN np array) is defined, will override 
    #     random sparse matrix generated by c
    def __init__(self, N, c, connectivity_matrix=None):
        super().__init__(N)
        self.connections = None
        if connectivity_matrix is not None:
            self.connections = connectivity_matrix
        else:
            self.connections = generate_sparse_connection_matrix(N, c)
        self.C = 0 # mean connections per node
        conn_count = np.sum(self.connections, axis=0)
        self.C = np.sum(conn_count)/N
        self.W = np.zeros((N,N))  

    def learn(self, data, labels):
        # first calculate learning threshold
        M = data.shape[0]
        a_arr = np.zeros(shape=self.N)
        for index in range(M):
            self.label_map[data[index].astype(int).tobytes()] = labels[index]
        a = 0.0
        for a_sub in range(self.N):
            a_arr_sum = 0.0
            for sub in range(M):
                if data[sub][a_sub] > 0:
                    a_arr_sum += 1
            a_arr[a_sub] = a_arr_sum / M
            a += a_arr[a_sub]
        a = a / self.N
        # would like to make this matrix algebra, for now do for loops
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                weight_sum = 0.0
                for k in range(M):
                    weight_sum += (data[k][i]) * (data[k][j] - a_arr[j])
                self.W[i][j] = weight_sum / (self.C * a)
                self.W[j][i] = self.W[i][j]

    def simulate(self, activation, beta=1, threshold=1, max_epoch=50):
        self.A = np.copy(activation).astype(np.float64)
        Aold = np.copy(activation).astype(np.float64)
        converged = False
        epochs = 0
        while not converged:
            print(self.A)
            epochs += 1
            # generate random ordering of nodes
            indices = np.random.permutation(self.N) 
            presynapse = np.copy(self.A) 
            for i in indices:                     
                hi = 0
                for j in range(self.N):
                    hi += self.connections[i][j]*self.W[i][j]*presynapse[j]
                if hi > 0:
                    self.A[i] = 1
                elif hi < 0:
                    self.A[i] = -1
            if all(self.A == Aold) or epochs > max_epoch:
                converged = True
            Aold = np.copy(self.A)
        print("Converged to", self.A)
        return self.label_map[self.A.astype(int).tobytes()]
    
    def sigmoidal(self, hi, beta, threshold):
        expo = beta*(threshold-hi)
        sigmoid = 1 / (1 + np.exp(expo))
        if sigmoid >= 0.5:
            return 1
        else:
            return 0 


