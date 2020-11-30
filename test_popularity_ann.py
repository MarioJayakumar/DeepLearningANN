from popularity_ann import PopularityANN, generate_sparse_connection_matrix
import numpy as np

full_connection_mat = np.ones(shape=(5,5)) - np.eye(5)
f = PopularityANN(N=5, c=3, connectivity_matrix=full_connection_mat)
print(f.connections)
print(f.C)

data = [[1,-1,-1,-1,-1],[1,1,1,-1,-1]]
data = np.array(data)
labels = ["e1", "e2"]

f.learn(data, labels)
print(f.W)
print(f.simulate(np.array([1,-1 ,1 ,-1, -1]), beta=1, threshold=0.5))
