import time
import numpy as np
from EDF_Percpetron import ReLU, Flattener
from Nueron_Layer import Input_Layer, Conv_layer, Linear_Softmax_Computation_Layer, Cross_Entropy, Linear_Computation_Layer

class CNN():
    ''' Combines all convulutional functionality into one class to reduce code clutter '''

    def __init__(self, n_features, n_outputs, architecture, test_node):
        # initialise atrributes
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.architecture = architecture
        self.graph = []

        # build graph
        self.input_layer = Input_Layer(self.n_features)
        self.graph.append(self.input_layer)

        # build convulutional layers
        for i in range(len(self.architecture) - 1):
            for _ in range(self.architecture[i][0] - 1):
                self.graph.append(Conv_layer(self.graph[-1], self.architecture[i][1], max_pooling=False))
            self.graph.append(Conv_layer(self.graph[-1], self.architecture[i][1], max_pooling=True))
        
        for _ in range(self.architecture[-1][0] -1):
            self.graph.append(Conv_layer(self.graph[-1], self.architecture[-1][1], max_pooling=False))

        self.graph.append(Conv_layer(self.graph[-1], self.architecture[-1][1], max_pooling=True))

        self.graph[-1].nodes.append(Flattener(self.graph[-1].nodes[-1]))

        self.graph.append(Linear_Computation_Layer(self.graph[-1], self.architecture[-1][1], ReLU))
        # build output layer
        self.output_layer = Linear_Softmax_Computation_Layer(self.graph[-1], self.n_outputs)
        self.test_node = test_node
        self.loss = Cross_Entropy(test_node, self.output_layer.nodes[-1])

        self.graph.extend([self.output_layer, self.loss])

        # Create graph outside the training loop
        self.trainable = self.graph[1:-1]
    
    def learn(self, x_train, y_train, epochs, batch_size, learning_rate, print_properties = False):
        # add values for model property recording
        start = time.time()
        losses = []

        for epoch in range(epochs):
            loss_value = 0
            for i in range(0,x_train.shape[0], batch_size):
                self.input_layer.value = x_train[i:min(i+batch_size, x_train.shape[0]), :, :, :]
                self.test_node.value = y_train[i:min(i+batch_size, y_train.shape[0]), :,]

                self.forward_pass()
                self.backward_pass()
                progress = (i + x_train.shape[0] * epoch)
                self.sgd_update(learning_rate, progress)

                loss_value += self.loss.value
                if print_properties: print(f"\r progress: {i+x_train.shape[0]*epoch}/{x_train.shape[0]*epochs} loss:{self.loss.value/batch_size}", end='', flush=True)

            if print_properties: print(f'\nEpoch: {epoch + 1}, loss: {loss_value / x_train.shape[0]}')
            losses.append(loss_value / x_train.shape[0])

        end = time.time()
        if print_properties: print(f"Size {batch_size}, Loss: {loss_value / x_train.shape[0]}\n Time: {(end - start):.4f} seconds")
        return losses, end-start

    def evaluate(self, x_test, y_test, evaluation_function, batch_size = 32):
        # evaluates the model against a set of test values
        correct_predictions = 0
        entropy = 0

        # rather than use one test value at a time, use a batch.
        for i in range(0,x_test.shape[0], batch_size):
            self.input_layer.value =  x_test[i:min(i+batch_size, x_test.shape[0]), :, :, :]
            self.test_node.value =  y_test[i:min(i+batch_size, y_test.shape[0]), :,]
            self.forward_pass()
            entropy += (-np.sum(self.output_layer.activation_node.value * np.log(self.output_layer.activation_node.value) ) )
            
            # Compare the largest value of each batch item to the corresponding hot_one value in test_y
            predictions = np.argmax(self.output_layer.activation_node.value, axis=1)
            correct_predictions += np.sum(predictions == np.argmax(y_test[i:min(i + batch_size, y_test.shape[0])], axis=1))
            
        avg_entropy = entropy / y_test.shape[0]
        accuracy = correct_predictions / x_test.shape[0]
        return accuracy, avg_entropy
    
    def forward_pass(self):
        for n in self.graph:
            n.forward()

    def backward_pass(self):
        for n in self.graph[::-1]:
            n.backward()

    # SGD Update
    def sgd_update(self, learning_rate=1e-2, progress=0):
        for t in self.trainable:
            t.grad_update(learning_rate / (1 + 0.2e-5 * progress))
