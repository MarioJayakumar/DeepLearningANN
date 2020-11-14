# Source code provide from CMSC727-Assignment 3 Files
# Modified for this project

import numpy as np
from collections import defaultdict
from attractor_net import AttractorNetwork

# Hamming dist between vectors a, b
def hamdist(a,b): return sum(a != b)  # Hamming dist vectors a, b

# Hopfield Network Class Definition
class hopnet(AttractorNetwork):

  def __init__ (self,N):              # create N-node Hopfield Net 
    self.A = np.zeros(N)                 # A = activity state
    self.W = np.zeros((N,N))             # W = N x N weight matrix
    self.E = 0                        # E = current network energy
    self.label_map = defaultdict(lambda:-1, {})
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
    
  # data should be M examples, where the ith example is vector of length N
  # so Data is MxN
  def learn(self,data,labels):          # learn matrix W from data	
    N = self.N
    M = data.shape[0]
    for i in range(M):         
      c_data = self.discretize_activation(data[i])     # for each input pattern i
      self.W = self.W + np.outer(c_data,c_data)
      self.label_map[c_data.tobytes()] = labels[i]
    self.W = (1.0 / N) * (self.W - M * np.eye(N,N))  # zeros main diag.

  def sgn(self,input,oldval):         # compute a = sgn(input)
    if input > 0: return 1
    elif input < 0: return -1
    else: return oldval

  def update(self):                       # asynchronously update A
    indices = np.random.permutation(self.N)  # determine order node updates
    for i in indices:                     # for each node i
      scalprod = np.dot(self.W[i,:],self.A)         # compute i's input
      self.A[i] = self.sgn(scalprod,self.A[i])   # assign i's act. value
    return

  def simulate(self,Ainit):
    Ainit = self.discretize_activation(Ainit)
    # Simulate Hopfield net starting in state Ainit.
    # Returns iteration number tlast and Hamming distance dist
    # of A from stored pattern Ainit when final state reached.
    # trace = 1 prints in fileid state A and energy E at each t
    t = 0                        # initialize time step t
    self.A = np.copy(Ainit)         # assign initial state A
    self.E = self.energy()       # compute energy E
    #fileid.write("self.A = {} \n".format(self.A))
    #self.showstate(fileid,t,self.E)
    Aold = np.copy(Ainit)           # A at previous t
    #fileid.write("Aold = {} \n".format(Aold))
    moretodo = True              # not known to be at equilibrium yet
    while moretodo:              # while fixed point not reached
      t += 1                     #   increment iteration counter
      self.update()              #   update all A values once per t
      #fileid.write("self.A after updating = {} \n".format(self.A))
      #fileid.write("Aold after updating = {} \n".format(Aold))
      self.E = self.energy()     #   compute energy E of state A
      #fileid.write("self.A after energy calc = {} \n".format(self.A))
      #fileid.write("Aold after energy calc = {} \n".format(Aold))
      if all(self.A == Aold):    #   if at fixed point
        #fileid.write("self.A == Aold satisfied\n")
        tlast = t                #      record ending iteration
        dist = hamdist(Ainit,self.A)    # distance from Ainit
        moretodo = False         #      and quit
      Aold = np.copy(self.A)
      #fileid.write("Aold after updating = {} \n".format(Aold))
    #fileid.write("after while termination: self.A = {}, Aold = {} \n".format(self.A,Aold))
    #self.showstate(fileid,t,self.E)
    predict_label = self.label_map[self.A.tobytes()]
    return predict_label

  def energy(self):             # Returns network's energy E 
    return -0.5 * np.dot(self.A,np.dot(self.W,self.A))

  def showstate(self,fileid,t,E):  # display A at time t
    fileid.write("t: {} ".format(t))
    for i in range(self.N):
      if self.A[i] == 1: fileid.write("+")
      else: fileid.write("-")
    fileid.write("  E: {} ".format(E))
    fileid.write("\n")

  def showwts(self,fileid):    # display W 
    fileid.write("\nWeights =\n")
    for i in range(self.N):
      for j in range(self.N):
        fileid.write(" {0:7.3f}".format(self.W[i,j]))
      fileid.write("\n")

