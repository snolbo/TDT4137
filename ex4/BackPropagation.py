import numpy as np


# Assuming output function is a sigmoid function
def train_weight(w_old, value, delta, alpha):
    return w_old + alpha * value * delta


def delta_k(y_k, error_k):
    return y_k*(1-y_k)*error_k


def delta_j(y_j, delta_k_vec, w_jk_vec):
    sum = 0
    for i in range(len(delta_k_vec)):
        sum += y_j*(1-y_j)*delta_k_vec[i]*w_jk_vec[i]
    return sum


def y_out(x):
    return 1/(1+np.exp(x))


def X_in(x_vec, w_vec, theta):
    sum = 0
    for i in range(0, len(w_vec)):
        sum += x_vec[i]*w_vec[i]
    return sum - theta


alpha = 0.1

# Inputs
x1 = 0
x2 = 1
# Desired output
yd5 = 0
yd6 = 1
Yd = [yd5, yd6]

# Weights
w13 = 0.5
w23 = 0.0
w3_in = [w13, w23]

w14 = 0.0
w24 = 0.9
w4_in = [w14, w24]

w35 = 0.4
w45 = -1.2
w5_in = [w35, w45]

w36 = 1.0
w46 = 1.1
w6_in = [w36, w46]


# Thresholds
theta3 = 0.8
theta4 = -0.1
theta5 = 0.3
theta6 = 0.5


print("Weights before iterations:")
print("w13: " + str(w13))
print("w14: " + str(w14))

print("w23: " + str(w23))
print("w24: " + str(w24))

print("w35: " + str(w35))
print("w36: " + str(w36))

print("w45: " + str(w45))
print("w46: " + str(w46))
print("Thresholds before iterations:")
print("theta3: " + str(theta3))
print("theta4: " + str(theta5))
print("theta5: " + str(theta5))
print("theta6: " + str(theta6))


# # FORWARD FEEDING
# Calculate sum at hidden layer nodes
Xi = [x1, x2]
n3_sum = X_in(Xi, w3_in, theta3)
n4_sum = X_in(Xi, w4_in, theta4)

# Produce output from hidden layer nodes
n3_y = y_out(n3_sum)
n4_y = y_out(n4_sum)

# Calculate sum at output layer nodes
Xj = [n3_y, n4_y]
n5_sum = X_in(Xj, w5_in, theta5)
n6_sum = X_in(Xj, w6_in, theta6)

# Produce output from output layer nodes
n5_y = y_out(n5_sum)
n6_y = y_out(n6_sum)
Y = [n5_y, n6_y]


# # BACKWARD PROPAGATION
# Calculate errors
e5 = 1/2*(yd5 - n5_y)**2
e6 = 1/2*(yd6 - n6_y)**2

# Calculate error gradient output layer and find new jk weights and thresholds
delta_k5 = delta_k(n5_y, e5)
delta_k6 = delta_k(n6_y, e6)
DELTA_K = [delta_k5, delta_k6]

w35_new = train_weight(w35, n3_y, delta_k5, alpha)
w45_new = train_weight(w45, n4_y, delta_k5, alpha)
theta5_new = train_weight(theta5, -1, delta_k5, alpha)

w36_new = train_weight(w36, n3_y, delta_k6, alpha)
w46_new = train_weight(w46, n4_y, delta_k6, alpha)
theta6_new = train_weight(theta6, -1, delta_k6, alpha)


# Calculate error gradient hidden layer and find new ij weights and thresholds
delta_j3 = delta_j(n3_y, DELTA_K, [w35, w36])
delta_j4 = delta_j(n4_y, DELTA_K, [w45, w46])

w13_new = train_weight(w13, x1, delta_j3, alpha)
w23_new = train_weight(w23, x2, delta_j3, alpha)
theta3_new = train_weight(theta3, -1, delta_j3, alpha)

w14_new = train_weight(w14, x1, delta_j4, alpha)
w24_new = train_weight(w24, x2, delta_j4, alpha)
theta4_new = train_weight(theta4, -1, delta_j4, alpha)


print("Weights after iterations:")
print("w13: " + str(w13_new))
print("w14: " + str(w14_new))

print("w23: " + str(w23_new))
print("w24: " + str(w24_new))

print("w35: " + str(w35_new))
print("w36: " + str(w36_new))

print("w45: " + str(w45_new))
print("w46: " + str(w46_new))

print("Thresholds before iterations:")
print("theta3_new: " + str(theta3_new))
print("theta4_new: " + str(theta5_new))
print("theta5_new: " + str(theta5_new))
print("theta6_new: " + str(theta6_new))









