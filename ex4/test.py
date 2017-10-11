import random
import matplotlib.pyplot as plt
import numpy


input1 = [0, 0, 1, 1]
input2 = [0, 1, 0, 1]

weights = []
weights.append(random.random() - 0.5)
weights.append(random.random() - 0.5)
#weights.append(0.1)
#weights.append(0.2)

thres = random.random() - 0.5
#thres = 0.2

streak = 0
alpha = 0.01


iterations = 0
print(str(weights) + str(thres))
while streak < 100 and iterations < 10000:
    # Find input
    c = random.randint(0, 3)
    in1 = input1[c]
    in2 = input2[c]

    # Calculate sum
    sum = 0
    sum += weights[0]*in1
    sum += weights[1]*in2
    sum -= thres

    #Evaluate results
    yd = in1 and in2
    y = sum > 0
    error = yd - y

    if error == 0:
        streak += 1
    else:
        streak = 0
    # print(streak)

    #Update weights
    weights[0] = weights[0] + alpha * in1 * error
    weights[1] = weights[1] + alpha * in2 * error
    thres = thres +  alpha * (-1) * error

    # Progress print
    iterations += 1



print(streak)
print("Trained iterations : " + str(iterations - 100))
print(str(weights) + str(thres))

