from EDF_Percpetron import *
import numpy as np


def linear_param_builder(w,input_layer_w):
    # the parameter factory used for the linear function
    return [Parameter(np.zeros((1, w))), 
            Parameter(np.random.randn(w, input_layer_w) * 0.1)]

def conv_param_builder(output_channels, input_channels):
    stddev = np.sqrt(0.5 / (input_channels))
    return [Parameter(np.zeros((output_channels))), 
            Parameter(np.random.randn(3, 3, input_channels, output_channels) * stddev)]

class Nueron_Layer:
    # The Nueron Layer base calss that input, and computational nodes are built upon

    def __init__(self, input_layer = None, width: int = 1,):
        self.input_layer = input_layer
        self.width = width
        self.output_layer = None
        self.value = None
        self.nodes = None
        if self.input_layer:
            self.input_layer.output_layer = self
    
    def forward(self):
        # perform forward pass
        raise NotImplementedError
    
    def backward(self):
        # perform backward pass
        raise NotImplementedError
    
class Input_Layer(Nueron_Layer):
    # The layer used for processing inputs
    def __init__(self, number_of_inputs):
        Nueron_Layer.__init__(self, input_layer= None, width= number_of_inputs)
        self.nodes = [Input()]
        
    def forward(self, value = None):
        self.value = value if value else self.value
        self.nodes[0].forward(self.value)
    
    def backward(self):
        self.nodes[0].backward()
    
class Computation_Layer(Nueron_Layer):
    # computation layers contain a nueron with an operation and activation equations, as well as trainable paramters
    # paramter factory is a function used to generate the parameter vectors according to specified width
    def __init__(self, input_layer=None, width = 1, operation: Node = Linear, activation: Node = Sigmoid, parameter_factory = linear_param_builder):
        Nueron_Layer.__init__(self, input_layer, width)
        self.paramter_nodes = parameter_factory(self.width, self.input_layer.width)
        self.operation_node = operation(self.paramter_nodes, self.input_layer.nodes[-1])
        self.activation_node = activation(self.operation_node)
        self.nodes = [*self.paramter_nodes, self.operation_node, self.activation_node]

    def forward(self):
        for n in self.nodes:
            n.forward()
        
    def backward(self):
        # backward and update the paramters
        for n in self.nodes[::-1]:
            n.backward()
        
    
    def grad_update(self, learning_rate = 0.001):
        for n in self.paramter_nodes:
            n.value = n.value - n.gradients[n] * learning_rate

class Linear_Computation_Layer(Computation_Layer):
    def __init__(self, input_layer=None, width=1, activation= Sigmoid):
        Computation_Layer.__init__(self, input_layer, width, Linear, activation, linear_param_builder)

class Linear_Softmax_Computation_Layer(Computation_Layer):
    def __init__(self, input_layer=None, width=1,):
        Computation_Layer.__init__(self, input_layer, width, operation=Linear, activation=Softmax, parameter_factory=linear_param_builder) 
    
class Conv_layer(Computation_Layer):
    def __init__(self, input_layer=None, output_features=1, activation = ReLU, max_pooling: bool = True):
        Computation_Layer.__init__(self, input_layer, output_features, Conv, activation, conv_param_builder)

        if max_pooling:
            self.nodes.append(MaxPooling(self.nodes[-1]))

    def grad_update(self, learning_rate = 0.001):
        for n in self.paramter_nodes:
            n.gradients[n] = np.clip(n.gradients[n], -10, 100)
            n.value = (n.value - n.gradients[n] * learning_rate)