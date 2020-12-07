from hiayn import Hopfield
import numpy as np

f = Hopfield(input_size=5)

data = [[1,0.3, 0.2,0.1,0.7]]
data = np.array(data)
labels = ["e1"]

f.learn(data, labels)
print(f.W)
print(f.threshold(data[0]))
print(f.simulate(np.array([1,-1,-1,-1,1])))
