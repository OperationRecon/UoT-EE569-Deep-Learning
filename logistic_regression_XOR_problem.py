from EDF_batch import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time

# Define constants
CLASS1_SIZE = 700
CLASS2_SIZE = 700
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.02
EPOCHS = 200
TEST_SIZE = 0.25
BATCH_SIZE = 4

# Define the means and covariances of the two components
MEAN1a = np.array([-3, -3])
COV1a = np.array([[1, 0], [0, 1]])
MEAN1b = np.array([3, 3])
COV1b = np.array([[1, 0], [0, 1]])
MEAN2a = np.array([-3, 3])
COV2a = np.array([[1, 0], [0, 1]])
MEAN2b = np.array([3, -3])
COV2b = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = np.vstack((multivariate_normal.rvs(MEAN1a, COV1a, int(np.floor(CLASS1_SIZE/2))),multivariate_normal.rvs(MEAN1b, COV1b, int(np.ceil(CLASS1_SIZE/2)))))
X2 = np.vstack((multivariate_normal.rvs(MEAN2a, COV2a, int(np.floor(CLASS2_SIZE/2))), multivariate_normal.rvs(MEAN2b, COV2b, int(np.ceil(CLASS2_SIZE/2)))))

# Combine the points and generate labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

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
n_features = X_train.shape[1]
n_output = 1

# initialize arrays to store evaluation results
times = []
fn_losses = []
accuracies = []


# Initialize weights and biases
W0 = np.zeros(1)
W1 = np.random.randn(1) * 0.1
W2 = np.random.randn(1) * 0.1

# Create nodes
x_node = Input()
y_node = Input()

w0_node = Parameter(W0)
w1_node = Parameter(np.array([W1, W2]))

# Build computation graph
lin_node = Linear(w1_node,x_node,w0_node)
sigmoid = Sigmoid(lin_node)
loss = BCE(y_node, sigmoid)

# Create graph outside the training loop
graph = [x_node, w0_node, w1_node, lin_node, sigmoid,loss]
trainable = [w0_node,w1_node,]

# Training loop
epochs = 100
learning_rate = 0.001

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
        for i in range(len(t.value)):
            t.value[i] -= learning_rate * np.average(t.gradients[t][i])


# Start a measure of the model's excecutrion speed
start = time.time()

for epoch in range(epochs):
    loss_value = 0
    for i in range(0,X_train.shape[0], BATCH_SIZE):
        x1 = X_train[i:min(X_train.shape[0], i + BATCH_SIZE), 0].reshape(1,-1)
        x2  = X_train[i:min(X_train.shape[0], i + BATCH_SIZE), 1].reshape(1, -1)
        x_node.value = np.vstack((x1,x2,))
        y_node.value = y_train[i:min(X_train.shape[0], i + BATCH_SIZE)].reshape(1, -1)

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, learning_rate)

        loss_value += loss.value

end = time.time()
times.append(end-start)
fn_losses.append(loss_value / X_train.shape[0])

print(f"Size {BATCH_SIZE}, Loss: {loss_value / X_train.shape[0]}\n Time: {(end - start):.4f} seconds")

# Evaluate the model
correct_predictions = 0
for i in range(X_test.shape[0]):
    x_node.value = np.vstack((X_test[i][0].reshape(-1, 1),X_test[i][1].reshape(-1, 1)))
    forward_pass(graph)

    if round(sigmoid.value[0][0]) == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
accuracies.append(accuracy)

print(f"Accuracy: {accuracy * 100:.3f}%")


x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = []
for i,j in zip(xx.ravel(),yy.ravel()):
    x_node.value = np.hstack((np.array([i]), np.array([j])))
    forward_pass(graph)
    Z.append(sigmoid.value)
Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()
