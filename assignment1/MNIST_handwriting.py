from sklearn import datasets
from EDF_Percpetron import *
from Nueron_Layer import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time
from MLP import MLP

# define constants
N_FEATURES = 64
N_OUTPUT = 10
LEARNING_RATE = 0.02
EPOCHS = 2000
TEST_SIZE = 0.4
BATCH_SIZE = 100
DEPTH = 1
WIDTH = 64


# evaluation function for "hot-one" outputs
def evaluation_function(x,y):
    return np.argmax(x) == y

# load data
mnist = datasets.load_digits()
X, y = mnist['data'], mnist['target'].astype(int)

hot_one_y = np.array([0] * 10 * y.shape[0]).reshape(y.shape[0], 10)

for i in range(y.shape[0]):
    hot_one_y[i][y[i]] = 1


# Split data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = hot_one_y[train_indices], y[test_indices]

y_node = Input()
mlp = MLP(N_FEATURES, N_OUTPUT, DEPTH, WIDTH,
        Linear_Computation_Layer, 
        Linear_Softmax_Computation_Layer, 
        Cross_Entropy,
        y_node)

print('learing...')
losses, learn_time = mlp.learn(X_train,y_train, EPOCHS, BATCH_SIZE, LEARNING_RATE,)
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}.\nLoss: {losses[-1]}, Learning time: {learn_time:.4f}.")


print('evaluating...')
accuracy, entropy = mlp.evaluate(X_test, y_test, evaluation_function)
print(f"Test size: {X_test.shape[0]}\nAccuracy: {accuracy*100:.4f}%, Average entropy: {entropy}")