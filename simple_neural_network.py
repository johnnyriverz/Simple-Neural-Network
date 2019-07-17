import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return np.exp(-x) / (pow((1 + np.exp(-x)),2))

# here I will include the different inputs in our data
training_input = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])

# here we will include the different outputs we got from the inputs above, this information is our data
training_output = np.array([[0,1,1,0]]).T # we add the T because we got one output per row

# we want to add random values to our weights
np.random.seed(1)

# we will create a 3x1 matrix since we have a matrix of 3 for inputs and a matrix of 1 for our outputs
synaptic_weights = (2 * np.random.random((3,1))-1)

# we want to record each results
print ("Random starting synaptic weights: ")
print (synaptic_weights)

# now our main loop
for iteration in range(20000):
  input_layer = training_input
  outputs = sigmoid(np.dot(input_layer, synaptic_weights))

# Now we review the outputs and calculate the error. This is the differance between the data output and the outputs after training.

  error = training_output - outputs

# now we calculate the weight adjustment

  adjustment = error * sigmoid_derivative(outputs)

# now adjust the synaptic weights accordingly

  synaptic_weights += np.dot(input_layer.T, adjustment)

print ("Synaptic weights after training: ")
print (synaptic_weights)
print ("Outputs after training: ")
print (outputs)

