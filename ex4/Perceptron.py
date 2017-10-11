import random
import matplotlib.pyplot as plt
import numpy


input1 = [0, 0, 1, 1]
input2 = [0, 1, 0, 1]

w = []
w.append(random.random() - 0.5)
w.append(random.random() - 0.5)
#w.append(0.1)
#w.append(0.2)

thres = random.random() - 0.5
#thres = 0.2

streak = 0
alpha = 0.05

import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-0.5, 1.5, 10)
y = [(-(w[0]/w[1]) * jj + thres/w[1]) for jj in x]

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([-0.5, 2])
ax.grid()
ax.scatter(input1, input2)

line1, = ax.plot(x, y, 'b-')

iterations = 0
print("Starting w : " + str(w) + " Starting treshold : " + str(thres))
while streak < 20 and iterations < 10000:
    # Find input
    c = random.randint(0, 3)
    in1 = input1[c]
    in2 = input2[c]

    # Calculate sum
    sum = 0
    sum += w[0]*in1
    sum += w[1]*in2
    sum -= thres

    #Evaluate results
    yd = in1 or in2
    y = sum > 0
    error = yd - y

    if error == 0:
        streak += 1
    else:
        streak = 0
    # print(streak)

    #Update w
    w[0] = w[0] + alpha * in1 * error
    w[1] = w[1] + alpha * in2 * error
    thres = thres + alpha * (-1) * error

    line1.set_ydata([-((w[0] / w[1]) * jj - (thres / w[1])) for jj in x])
    fig.canvas.draw()
    plt.pause(0.01)

    # Progress print
    iterations += 1



print()
print("Trained iterations : " + str(iterations - 20) + " Before convergence")
print("Ending w : " + str(w) + " Ending treshold : " + str(thres))

