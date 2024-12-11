import numpy as np
from collections.abc import Iterable

# Base Node class
class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}

        for node in inputs:
            node.outputs.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


# Input Node
class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

# Parameter Node
class Parameter(Node):
    def __init__(self, value):
        Node.__init__(self)
        self.value = value

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

class Multiply(Node):
    def __init__(self, x, y):
        # Initialize with two inputs x and y
        Node.__init__(self, [x, y])

    def forward(self):
        # Perform element-wise multiplication
        x, y = self.inputs
        self.value = x.value * y.value

    def backward(self):
        # Compute gradients for x and y based on the chain rule
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self] * y.value
        self.gradients[y] = self.outputs[0].gradients[self] * x.value

class Addition(Node):
    def __init__(self, x, y):
        # Initialize with two inputs x and y
        Node.__init__(self, [x, y])

    def forward(self):
        # Perform element-wise addition
        x, y = self.inputs
        self.value = x.value + y.value

    def backward(self):
        # The gradient of addition with respect to both inputs is the gradient of the output
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self]
        self.gradients[y] = self.outputs[0].gradients[self]

# Linear equation Node (param[0] * var + param[1])
class Linear(Node):
    def __init__(self, param, var):
        Node.__init__(self, [*param, var])

    def forward(self):
        # Perform forward-pass (addidtion plus multiplication)
        b, a, x = self.inputs
        
        mul = np.dot(np.transpose(a.value), x.value)
        self.value = mul + b.value

    def backward(self):
        b, a, x = self.inputs

        # the gradient of b is the gradient of the output
        self.gradients[b] = self.outputs[0].gradients[self]

        # whereas the gradient of a and x follows the chain rule

        self.gradients[x] = np.dot(a.value, self.outputs[0].gradients[self],)
        self.gradients[a] = self.outputs[0].gradients[self][np.newaxis,:,:] * x.value[:,np.newaxis,:]

        
# Sigmoid Activation Node
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]

class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        self.value = np.sum(-y_true.value*np.log(y_pred.value)-(1-y_true.value)*np.log(1-y_pred.value))

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (1 / y_true.value.shape[0]) * (y_pred.value - y_true.value)/(y_pred.value*(1-y_pred.value))
        self.gradients[y_true] = (1 / y_true.value.shape[0]) * np.log(y_pred.value) - np.log(1-y_pred.value)

class Softmax(Node):
    def __init__(self, node=None):
        super().__init__([node])

    def _softamx(self, x):
        # does the softmax operation, apparently reduceing all entries by the largest one is used to improve stablity.
        ex = np.exp(x - np.max(x, axis= 0, keepdims=True))
        return ex / np.sum(ex, axis = 0, keepdims=True)
    
    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._softamx(input_value)
    
    def backward(self):
        # when coupling the Softmax funtion with a cross-entropy loss function, apprently that causes the gradient to simplify into a much more managable term. here we will use that propertey to our advantage
        self.gradients[self.inputs[0]] = self.value - self.outputs[0].gradients[self]

class Cross_Entropy(Node):
    def __init__(self, y_true, y_pred):
        super().__init__([y_true, y_pred])
    
    def forward(self):
        y_true, y_pred = self.inputs
        self.value = -np.sum(y_true.value * np.log(y_pred.value))
    
    def backward(self):
        # depending on the softmax coubling drevative simplfication. we simply pass back the true values
        y_true, y_pred = self.inputs
        self.gradients[y_true] = y_pred.value
        self.gradients[y_pred] = y_true.value