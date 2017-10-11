from pybrain3.tools.shortcuts import buildNetwork


# 2 input, 3 hidden and 1 output - node
net = buildNetwork(2, 3, 1)
net.activate([2, 1])