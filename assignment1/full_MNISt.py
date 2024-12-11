import numpy as np
from MLP import MLP
from EDF_Percpetron import *
from Nueron_Layer import Linear_Computation_Layer, Linear_Softmax_Computation_Layer
# For some reason, VSC refuses to acknowledge that I have Keras. so trying to run this code via VSC throws "Module Not Found" error, whereas using the shell runs the programm carefully.
import keras
from matplotlib import pyplot

#loading the dataset
(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()
(train_X, train_y), (test_X, test_y) = (train_X[:2000], train_y[:2000]), (test_X[:100], test_y[:100])
#printing the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

train_X = train_X.reshape(-1,28*28)
test_X = test_X.reshape(-1,28*28)

# reshaping the MNIST deataset to fit into our model (hopefully)
print('X_train: ' + str(train_X.shape))
print('X_test:  '  + str(test_X.shape))

# define constants
N_FEATURES = 784
N_OUTPUT = 10
LEARNING_RATE = 0.02
EPOCHS = 1000
TEST_SIZE = 0.4
BATCH_SIZE = 100
DEPTH = 3
WIDTH = 128

# evaluation function for "hot-one" outputs
def evaluation_function(x,y):
    return np.argmax(x) == y

hot_one_y = np.array([0] * 10 * train_y.shape[0]).reshape(train_y.shape[0], 10)

for i in range(train_y.shape[0]):
    hot_one_y[i][train_y[i]] = 1

y_node = Input()
mlp = MLP(N_FEATURES, N_OUTPUT, DEPTH, WIDTH,
        Linear_Computation_Layer, 
        Linear_Softmax_Computation_Layer, 
        Cross_Entropy,
        y_node)

print('learing...')
losses, learn_time = mlp.learn(train_X,hot_one_y, EPOCHS, BATCH_SIZE, LEARNING_RATE,print_properties=True)
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}.\nLoss: {losses[-1]}, Learning time: {learn_time:.4f}.")


print('evaluating...')
accuracy, entropy = mlp.evaluate(train_X, train_y, evaluation_function)
print(f"Test size: {test_X.shape[0]}\nAccuracy: {accuracy*100:.4f}%, Average entropy: {entropy}")
