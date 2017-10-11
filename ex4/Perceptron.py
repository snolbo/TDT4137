import random

class Xi:
    def __init__(self, neuron):
        self.neuron = neuron
        self.input_is_set = False

        self.received_input = 0
        self.weight = 0# random.random() - 0.5

    def set_input(self, val):
        self.received_input = val
        if self.neuron is not None:
            self.neuron.receive_input(self)
        return





class Perceptron:

    learning_rate = 0.01

    def __init__(self, id):
        self.id_ = id
        self.num_inputs_ = 0

        self.Xi_list = []
        self.Yi_list = []

        self.current_input_sum_ = 0
        self.current_num_inputs_received = 0

        self.threshold_ = random.random() - 0.5

        return

    def add_input_source_(self):
        self.num_inputs_ += 1
        xi = Xi(self)
        self.Xi_list.append(xi)
        return xi

    def connect_to_output_source(self, output_source):
        yi = output_source.add_input_source_(self)
        self.Yi_list.append(yi)
        return

    def receive_input(self, xi):
        self.current_input_sum_ += xi.weight * xi.received_input
        self.current_num_inputs_received += 1
        # print("Input: " + str(xi.received_input) + "Weight: " + str(xi.weight))
        if self.current_num_inputs_received == self.num_inputs_:
            print("input sum " + str(self.current_input_sum_))
            output_result = self.current_input_sum_ > 0  # ???????????? STEP FUNCTION???
            self.current_input_sum_ = 0
            self.current_num_inputs_received = 0
            self.send_output_(output_result)
        return

    def send_output_(self, val):
        # print("Sending output")
        if len(self.Yi_list) == 0:
            print("Output from node " + str(self.id_) + "= " + str(val))
        for yi in self.Yi_list:
            yi.set_input(val)
        return val

    def train_weights(self, error):
        i = 0
        for xi in self.Xi_list:
            new_weight = xi.weight + Perceptron.learning_rate * xi.received_input * error - self.threshold_
            print("xi_id: " + str(i))
            print(" Old weight: " + str(xi.weight))
            print(" new weight: " + str(new_weight))
            xi.weight = new_weight
            i += 1
            # What about thresholds??????????
        print()
        return





# Create neuron, give it id = 0
p11 = Perceptron(0)



# Create input nodes to give data to, p11 is set as neuron to receive input when set
x1 = p11.add_input_source_()
x2 = p11.add_input_source_()

# Create output node to read data from
y1 = Xi(None)

# Create combinations of data input
i1 = [0, 0, 1, 1]
i2 = [0, 1, 0, 1]

# add y1 as output source, just pass 0 as key here, does not matter. key is meant to be pointer to another neuron
p11.Yi_list.append(y1)

correct_streak = 0
iterations = 0
while correct_streak < 1000:
    print("correct_streak = " + str(correct_streak))
    # Choose input combination
    index = random.randint(0, 3)
    # Activates input at p11
    x1.set_input(i1[index])
    x2.set_input(i2[index])
    # Find desired result
    Yd = i1[index] or i2[index]
    # Get calculated result
    Y = y1.received_input
    # Calculate error
    error = Yd - Y

    if error == 0:
        correct_streak += 1
    else:
        correct_streak = 0
    print(" Error " + str(error))

    # Train weights
    p11.train_weights(error)
    iterations += 1

print(str(iterations))







