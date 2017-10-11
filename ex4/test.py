import random

i1 = [0, 0, 1, 1]
i2 = [0, 1, 0, 1]



weights = []
weights.append(random.random() - 0.5)
weights.append(random.random() - 0.5)

thres = random.random() - 0.5


streak = 0

alpha = 0.01

iterations = 0
while streak < 1000:
    c = random.randint(0, 3)
    in1 = i1[c]
    in2 = i2[c]

    sum = 0
    sum += weights[0]*in1  - thres
    sum += weights[1]*in2  - thres

    yd = in1 and in2
    y = sum > 0
    error = yd - y

    if error == 0:
        streak +=1
    else:
        streak = 0
    # print(streak)
    weights[0] = weights[0] + alpha * in1 * error
    weights[1] = weights[1] + alpha * in2 * error
    iterations += 1
    if iterations % 100000 == 0:
        print(iterations)

print("Trained iterations : " + str(iterations - 1000))

