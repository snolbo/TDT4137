## TASK IS TO CREATE AN AUTO_ENCODER, A NETWORK WHERE INPUT EQUALS OUTPUT
# By having less dimentions/doing some simplifications, this can give results
# in the same way principal component analysis PCA - does, but only outputing
# the most characteristic properties of the input
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer


ds = SupervisedDataSet(1,1)
for i in range(1, 9):
    ds.addSample(i, i)

print ds



for nodes in range(8, 0, -1):
    print "\n TESTING NET WITH " + str(nodes) + " NODES"
    # one node in input, 8 in hidden, 1 in output
    net = buildNetwork(1, nodes, 1, hiddenclass=TanhLayer)
    # print net
    # print net['in']
    # print net['hidden0']
    # print net['out']

    trainer = BackpropTrainer(net, ds)
    errors = trainer.trainUntilConvergence(verbose=False, validationProportion=0.15, maxEpochs=1000, continueEpochs=10)
    # print errors

    test_data_trained = [x for x in range(1,9)]
    test_data_untrained = [x for x in range(10,20)]
    test_data_negative = [x for x in range(-10, 0)]

    print "Testing netowrk on data is has been trained for"
    print "Testing on: " + str(test_data_trained)
    res1 = [net.activate([val])[0] for val in test_data_trained]
    res2 = [net.activate([val])[0] for val in test_data_untrained]
    res3 = [net.activate([val])[0] for val in test_data_negative]
    print "Result on trained:"
    print "test set: " + str(test_data_trained) + ". Results: " + str(res1)
    print "test set: " + str(test_data_untrained) + ". Results: " + str(res2)
    print "test set: " + str(test_data_negative) + ". Results: " + str(res3)

