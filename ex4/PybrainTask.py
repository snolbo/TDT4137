## TASK IS TO CREATE AN AUTO_ENCODER, A NETWORK WHERE INPUT EQUALS OUTPUT
# By having less dimentions/doing some simplifications, this can give results
# in the same way principal component analysis PCA - does, but only outputing
# the most characteristic properties of the input
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from random import shuffle

from random import randint

# Function to print the weight between nodes of a net 100% copy paste.
def pesos_conexiones(n):
    for mod in n.modules:
        for conn in n.connections[mod]:
            print conn
            for cc in range(len(conn.params)):
                print conn.whichBuffers(cc), conn.params[cc]


train_data = []
ds = SupervisedDataSet(1,1)
for i in range(1, 9):
    # val = randint(1,8)  # random values
    val = i  # values between 1-8
    train_data.append(val)
replicate_train_data = []

# Create multiple examples from data
for i in range(0,5):
    for val in train_data:
        ds.addSample(val,val)
        replicate_train_data.append(val)
# Shuffle data
shuffle(replicate_train_data)
# Create other test.tiff data
test_data_untrained = [x for x in range(10,20)]
test_data_negative = [x for x in range(-10, 0)]

# print ds

# Test for different number of hidden nodes
for nodes in range(8, 0, -1):
    print "\n TESTING NET WITH " + str(nodes) + " NODES"
    net = buildNetwork(1, nodes, 1, hiddenclass=TanhLayer)
    # pesos_conexiones(net)
    # Create trainer and train the network
    trainer = BackpropTrainer(net, ds)
    errors = trainer.trainUntilConvergence(verbose=False, validationProportion=0.15, maxEpochs=1000, continueEpochs=10)
    # pesos_conexiones(net)

    # Get results from different test.tiff sets
    res1 = [net.activate([val])[0] for val in replicate_train_data]
    res2 = [net.activate([val])[0] for val in test_data_untrained]
    res3 = [net.activate([val])[0] for val in test_data_negative]
    print "test.tiff set: " + str(replicate_train_data) + ". Results: " + str(res1)
    print "test.tiff set: " + str(test_data_untrained) + ". Results: " + str(res2)
    print "test.tiff set: " + str(test_data_negative) + ". Results: " + str(res3)

    # Find the error in the result for when using train-data as test.tiff-data
    error = 0
    for i in range(0,len(replicate_train_data)):
        error += pow(res1[i]-replicate_train_data[i], 2)
    error = error / float(len(replicate_train_data))
    print "Sum error squared divided by testcases for training set: " + str(error)


