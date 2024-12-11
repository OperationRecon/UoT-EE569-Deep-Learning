import time
import numpy as np
from Nueron_Layer import Input_Layer


class MLP():
    # Combines all MLP functionality into one class to reduce code clutter
    def __init__(self, n_features, n_outputs, depth, width, hidden_layer, output_layer, loss_node_type, test_node):
        # initialise atrributes
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.depth = depth # Number of Hidden Layers
        self.width = width
        self.graph = []

        # build graph
        self.input_layer = Input_Layer(self.n_features)
        self.graph.append(self.input_layer)

        # build hidden layers
        for i in range(1,self.depth+1):
            self.graph.append(hidden_layer(self.graph[i-1], self.width,))

        # build output layer
        self.output_layer = output_layer(self.graph[-1], self.n_outputs)
        self.test_node = test_node
        self.loss = loss_node_type(test_node, self.output_layer.nodes[-1])

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
                self.input_layer.value = np.vstack([x[i:min(x_train.shape[0], i + batch_size)] for x in x_train.transpose()])
                self.test_node.value = np.vstack([x[i:min(y_train.shape[0], i + batch_size)] for x in y_train.transpose()])

                self.forward_pass()
                self.backward_pass()
                self.sgd_update(learning_rate)

                loss_value += self.loss.value

            if print_properties: print(f'Epoch: {epoch + 1}, loss: {loss_value / x_train.shape[0]}')
            losses.append(loss_value / x_train.shape[0])

        end = time.time()
        if print_properties: print(f"Size {batch_size}, Loss: {loss_value / x_train.shape[0]}\n Time: {(end - start):.4f} seconds")
        return losses, end-start

    def evaluate(self, x_test, y_test, evaluation_function):
        # evaluates the model against a set of test values
        correct_predictions = 0
        entropy = 0
        for i in range(x_test.shape[0]):
            self.input_layer.value = np.vstack(x_test[i])
            self.forward_pass()
            entropy += (-np.sum(self.output_layer.activation_node.value * np.log(self.output_layer.activation_node.value) ) )
            if evaluation_function(self.output_layer.activation_node.value, y_test[i]):
                correct_predictions += 1
            
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
    def sgd_update(self, learning_rate=1e-2):
        for t in self.trainable:
            t.grad_update(learning_rate)
