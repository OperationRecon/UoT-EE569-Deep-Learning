from EDF_Percpetron import *
from MLP import MLP
from Nueron_Layer import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time

# Define constants
CLASS1_SIZE = 300
CLASS2_SIZE = 300
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.5
EPOCHS = 8000
TEST_SIZE = 0.35
BATCH_SIZE = int((CLASS1_SIZE + CLASS2_SIZE) * TEST_SIZE)
DEPTH = 2
WIDTH = 20

# Define the means and covariances of the two components
MEAN1a = np.array([-2, -4])
COV1a = np.array([[1, 0], [0, 1]])
MEAN1b = np.array([3, 2])
COV1b = np.array([[1, 0], [0, 1]])
MEAN2a = np.array([0, 0])
COV2a = np.array([[1, 0], [0, 1]])
MEAN2b = np.array([2, -4])
COV2b = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = np.vstack((multivariate_normal.rvs(MEAN1a, COV1a, int(np.floor(CLASS1_SIZE/2))),multivariate_normal.rvs(MEAN1b, COV1b, int(np.ceil(CLASS1_SIZE/2)))))
X2 = np.vstack((multivariate_normal.rvs(MEAN2a, COV2a, int(np.floor(CLASS2_SIZE/2))), multivariate_normal.rvs(MEAN2b, COV2b, int(np.ceil(CLASS2_SIZE/2)))))

# Combine the points and generate labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE))).reshape(-1,1)

# Plot the generated data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
plt.show()

# Split data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]



# Model parameters

# initialize arrays to store evaluation results
times = []
fn_losses = []
accuracies = []


# Create nodes
y_node = Input()

mlp = MLP(N_FEATURES, N_OUTPUT, DEPTH, WIDTH, Linear_Computation_Layer,
          Linear_Computation_Layer, BCE, y_node)

graph = mlp.graph
loss = mlp.loss
trainable = mlp.trainable
input_layer = mlp.input_layer
output_layer = mlp.output_layer

# Training loop
epochs = EPOCHS
learning_rate = LEARNING_RATE

# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

# SGD Update
def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.grad_update(learning_rate)

losses, learn_time = mlp.learn(X_train, y_train, epochs, BATCH_SIZE, learning_rate)

print(f"Size {BATCH_SIZE}, Loss: {losses[-1] / X_train.shape[0]}\n Time: {learn_time:.4f} seconds")

# Evaluate the model
def binary_classifier_evaluator(output_layer_value, test_value):
    return round(output_layer_value[0][0]) == test_value

accuracy = mlp.evaluate(X_test, y_test, binary_classifier_evaluator)

print(f"Accuracy: {accuracy * 100:.3f}%")


x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = []
for i,j in zip(xx.ravel(),yy.ravel()):
    input_layer.value = np.vstack((np.array([i]), np.array([j])))
    forward_pass(graph)
    Z.append(output_layer.activation_node.value)
Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()

plt.plot(range(1,EPOCHS+1), losses,)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Loss Vs. Batch Size')
plt.show()

