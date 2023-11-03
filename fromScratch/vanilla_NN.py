import numpy as np

"""
	Function to initialize weights and biases.
	Here hidden1_size is the number of neurons in 1st hidden layer which is 4
	hidden2_size is the number of neurons in 2nd hidden layer which is 2
	output_size is number of neurons in output, since it is binary classification we have only 1 neuron at the end (output layer)
"""
def initialize_weights_and_biases(input_size, hidden1_size, hidden2_size, output_size):
		
		#first set of weights for layer 1
    w1 = np.random.rand(input_size, hidden1_size)
    b1 = np.random.rand(1, hidden1_size)

		#second set of weights for layer 2
    w2 = np.random.rand(hidden1_size, hidden2_size)
    b2 = np.random.rand(1, hidden2_size)

		#third set of weights for layer 3
    w3 = np.random.rand(hidden2_size, output_size)
    b3 = np.random.rand(1, output_size)

    return w1, b1, w2, b2, w3, b3

# Example usage
input_size = 3
hidden1_size = 4
hidden2_size = 2
output_size = 1

# Initialize weights and biases
w1, b1, w2, b2, w3, b3 = initialize_weights_and_biases(input_size, hidden1_size, hidden2_size, output_size)
# Generate random input
x = np.random.rand(1, input_size)


"""
After the above computation now the shapes are like the following:
x = (1,3) because there are 3 inputs
w1 = (3,4) 
b1 = (1,4)

w2 = (4,2)
b2 = (1,2)

w3 = (2,1) here 1 becasue the output is just 1 neuron
b3 = (1,1)
"""


# Function to create the neural network
def neural_network(x, w1, b1, w2, b2, w3, b3):

    z1 = np.dot(x, w1) + b1 #output of z1 is a (1,4) matrix
    a1 = 1/(1+np.exp(-z1))
    
		# Here now the a1 which is (1,4) and w2 is (4,2). 2 because the next layer has 2 neurons
    z2 = np.dot(a1,w2) + b2 #output of z2 is a (1,2) matrix
    a2 = 1/(1+np.exp(-z2))
    
		# Here now the a2 which is (1,2) and w3 is (2,1). 1 because the output layer has only 1 neuron
    z3 = np.dot(a2,w3) + b3 #output of z3 is a (1,1) matrix
    a3 = 1/(1+np.exp(-z3))

    return a3

# Forward pass through the neural network
model_output = neural_network(x, w1, b1, w2, b2, w3, b3)


