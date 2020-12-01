from orthogonal_hebbs_ann import OrthogonalHebbsANN
import numpy as np

f = OrthogonalHebbsANN(N=5)

data = [[1,-1,-1,-1,-1],[1,1,1,-1,-1], [1,-1,1,-1,1]]
data = np.array(data)
labels = ["e1", "e2", "e3"]

f.learn(data, labels)
print(f.W)
print(f.simulate(np.array([1,1 ,1 ,1, -1])))
