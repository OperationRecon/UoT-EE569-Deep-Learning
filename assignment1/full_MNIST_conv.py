import numpy as np
from CNN import CNN
from EDF_Percpetron import *
from Nueron_Layer import Linear_Computation_Layer, Linear_Softmax_Computation_Layer
# For some reason, VSC refuses to acknowledge that I have Keras. so trying to run this code via VSC throws "Module Not Found" error, whereas using the shell runs the programm carefully.
from tensorflow import keras
from matplotlib import pyplot

#loading the dataset
(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()

train_X = train_X / 255
train_X = np.expand_dims(train_X, axis=3)
test_X = test_X / 255
test_X = np.expand_dims(test_X, axis=3)



#printing the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))




for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
# pyplot.show()


# define constants
N_FEATURES = 1
N_OUTPUT = 10
LEARNING_RATE = 3.2
LEARNING_RATE_DECAY = 10
EPOCHS = 14
BATCH_SIZE = 150
ARCHITECTURE = [(1,4),(1,16),(1,32),(1,64),(1,128)]


hot_one_y = np.zeros((train_y.size, train_y.max() + 1))
hot_one_y[np.arange(train_y.size), train_y] = 1

#re-printing the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(hot_one_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

y_node = Input()
cnn = CNN(N_FEATURES, N_OUTPUT,
        ARCHITECTURE,
        y_node)

print('learing...')
losses, learn_time = cnn.learn(train_X,hot_one_y, EPOCHS, BATCH_SIZE, LEARNING_RATE, LEARNING_RATE_DECAY, print_properties=True)

with open("full_mnist_conv_evaluations.txt", 'a') as f:
    print(f"Architecture: {ARCHITECTURE}\nLearning rate: {LEARNING_RATE}\nBatch size: {BATCH_SIZE}, Epochs: {EPOCHS}.\nLoss: {losses[-1]}, Learning time: {learn_time:.4f}.", file=f)

print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}.\nLoss: {losses[-1]}, Learning time: {learn_time:.4f}.",)



print('evaluating...')
hot_one_y = np.zeros((test_y.size, test_y.max() + 1))
hot_one_y[np.arange(test_y.size), test_y] = 1
accuracy, entropy = cnn.evaluate(test_X, hot_one_y, BATCH_SIZE*2)

with open("full_mnist_conv_evaluations.txt", 'a') as f:
    print(f"Test size: {test_X.shape[0]}\nAccuracy: {accuracy*100:.4f}%, Average entropy: {entropy}", file=f)

print(f"Test size: {test_X.shape[0]}\nAccuracy: {accuracy*100:.4f}%, Average entropy: {entropy}",)
