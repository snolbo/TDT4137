
class Set:
    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1
        return

    def membership_of(self, x):
        if self.x0 >  x or self.x1 < x:
            return 0
        else:
            return 1

class TriangleSet(Set):
    def __init__(self, x0, x1, x_top):
        super(TriangleSet, self).__init__(x0, x1)
        self.x_top = x_top

    def membership_of(self, x):
        if self.x0 > x or self.x1 < x:
            return 0
        else:
            if x <= self.x_top and x >= self.x0:
                return (x-self.x0) / (self.x_top - self.x0)
            else:
                return (self.x1 - x) / (self.x1 - self.x_top)


class Grade(Set):
    def __init__(self, x0, x1):
        super(Grade, self).__init__(x0, x1)

    def membership_of(self, x):
        if x >= self.x1:
            return 1
        elif x <= self.x0:
            return 0
        else:
            return (x-self.x0)/(self.x1-self.x0)


class ReverseGrade(Set):
    def __init__(self, x0, x1):
        super(ReverseGrade, self).__init__(x0, x1)

    def membership_of(self, x):
        if x >= self.x1:
            return 0
        elif x <= self.x0:
            return 1
        else:
            return (self.x1-x)/(self.x1-self.x0)


## Creating sets
# distance
very_small = ReverseGrade(1, 2.5)
small = TriangleSet(1.5, 4.5, 3)
perfect = TriangleSet(3.5, 6.5, 5)
big = TriangleSet(5.5, 8.5, 7)
very_big = Grade(7.5, 9.0)

# delta
shrinking_fast = ReverseGrade(-4, -2.5)
shrinking = TriangleSet(-3.5,-0.5, -2)
stable = TriangleSet(-1.5, 1.5, 0)
growing = TriangleSet(0.5, 3.5, 2)
growing_fast = Grade(2.5, 4)

# action
brake_hard= ReverseGrade(-8, -5)
slow_down = TriangleSet(-7, -1, 4)
none = TriangleSet(-3, 3, 0)
speed_up = TriangleSet(1, 7, 4)
floor_it = Grade(5, 8)



# Values to evaluate
delta_val = 1.2
distance_val = 3.7

# Rule aggregation and memberships
brake_hard_membership = very_small.membership_of(distance_val)
slow_down_membership = min(small.membership_of(distance_val), stable.membership_of(delta_val))
none_membership = min(small.membership_of(distance_val), growing.membership_of(delta_val))
speed_up_membership = min(perfect.membership_of(distance_val), growing.membership_of(delta_val))
floor_it_membership = min(very_big.membership_of(distance_val), max(1-growing.membership_of(delta_val), 1-growing_fast.membership_of(delta_val)))

# Rule aggregation results
print("Brake hard: " + str(brake_hard_membership))
print("Slow Down: " + str(slow_down_membership))
print("None : " + str(none_membership))
print("Speed up: " + str(speed_up_membership))
print("Floor it: " + str(floor_it_membership))


# Preparation for processing
action = [brake_hard, slow_down, none, speed_up, floor_it]
action_membership = [brake_hard_membership, slow_down_membership, none_membership, speed_up_membership, floor_it_membership]
action_string_representation = ["BrakeHard", "SlowDown", "None", "SpeedUp", "FloorIt"]

# Defuzzification
interval = [-10, 10]
COG = 0
weights = 0
for val in range(interval[0], interval[1] +1): # For every val in interval
    max_membership = 0
    for act_index in range (0, len(action)): # Find max membership ( OR SHOULD IT BE SUMMATED??)
        if action[act_index].membership_of(val) > 0: # This value has membership in the set
            max_membership = max(max_membership, action_membership[act_index]) #decide from which set is has meximum membership
    # add value to weighted sum and sum of weights
    COG += val*max_membership
    weights += max_membership
# Calculate actual COG
COG = COG / weights

# Find which set this COG has memberships in
decided_action = []
for act in action:
    decided_action.append(act.membership_of(COG))


# Printing results
print()
print("COG: " + str(decision))
print("Chosen action: " + action_string_representation[decided_action.index(max(decided_action))]) # max of memberships is decision

