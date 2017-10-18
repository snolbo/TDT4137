## BUILDING NETWORKS
from pybrain.tools.shortcuts import buildNetwork
# 2 input, 3 hidden and 1 output - node in the three layers
net = buildNetwork(2, 3, 1)

# calculate output given the input,
# expect a tupple or array as input
output = net.activate([2,1])
print output

from pybrain.structure import TanhLayer
# Creates a net where the hidden layer is buildt with tanh istead of sigmoid
net2 = buildNetwork(2, 3, 1, hiddenclass=TanhLayer)
# prints the activation function used in the first hidden layer
print net2['hidden0']



from pybrain.structure import SoftmaxLayer
net3 = buildNetwork(2, 3, 2, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
output3 = net.activate((2,3))
print output3

# Tell the network to sue bias
net4 = buildNetwork(2, 3, 1, bias=True)
print net4['bias']


## BUILDING DATASET
from pybrain.datasets import SupervisedDataSet
# This datasets support 2D input and 1D target
ds = SupervisedDataSet(2, 1)


## Adding samples
# Adding xor function
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

# Examining dataset
print "Size of dataset: " + str(len(ds))
# iterate over dataset
print "Printing dataset:"
for input, target in ds:
    print input, target

#Get input and target as arrays
print "Accessing input and target as using dicts"
print ds['input']
print ds['target']

# # Clear dataset and deleta all values:
# print "Clear dataset and try to access input and target again"
# ds.clear()
# print ds['input']
# print ds['target']



## TRAINING NETWORK ON DATASET
from pybrain.supervised.trainers import BackpropTrainer
net5 = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
trainer = BackpropTrainer(net, ds)

# Trains the net for one full epoch and returns double proportional to error
error = trainer.train()
print "Error from training network one epoch: " + str(error)

# Trains the network until convergence, return errors from all epocs
error_tupple = trainer.trainUntilConvergence()
print "Errors from training the network until convergence"
print error_tupple




