import numpy as np
def NN(m1, m2, w1, w2, b):
    z = m1 * w1 + m2 * w2 + b
    return sigmoid(z)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

w1 = np.random.rand()
w2 = np.random.rand()
b = np.random.rand()

NN(2, 1, w1, w2, b)

''''-----------------------------------------------------------------------------'''

def cost(b):
    return (b-4) ** 2

cost(4)

def num_slope(b):
    h = 0.0001
    return (cost(b+h) - cost(b))/h

num_slope(3)
num_slope(5)

def slope(b):
    return 2 * (b - 4) 

slope(3)
slope(5)

b = 8                     # you can initiate -20 also                                
b = b - 0.1 * slope(b)   # if you run this multiple times the value gets closer to 4
print(b)

# Training Loop
for i in range(100):
    b = b - 0.1 * slope(b) 
    

'''------------------------------------------------------------------------------'''

import numpy as np
import matplotlib.pyplot as plt

# Each point is length, width, type(0 or 1)
#0 --- blue
#1 --- red


# each point is length, width, type (0, 1)

data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, .5,  1],
        [2,   .5,  0],
        [5.5,  1,  1],
        [1,    1,  0]]

mystery_flower = [4.5, 1]

# network

#       o  flower type
#      / \  w1, w2, b
#     o   o  length, width



























