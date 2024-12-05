from EDF_Percpetron import *
import numpy as np

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
    def __init__(self, input_layer=None, width = 1, operation: Node = Linear, activation: Node = Sigmoid, parameter_factory = None):
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
            
            # So numpy doesn't like the way the parameter arrays are setup, the try is for the b parameters while the except is for the a parameters.
            try:
                avg = np.average(n.gradients[n],axis=len(n.gradients[n].shape)-1, keepdims=True)
                tmp = learning_rate * avg
                n.value -= tmp

            except ValueError: 
                avg = np.average(n.gradients[n],axis=len(n.gradients[n].shape)-1, keepdims=False)
                tmp = learning_rate * avg
                n.value -= tmp
