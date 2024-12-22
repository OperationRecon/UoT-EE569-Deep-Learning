import numpy as np

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


# Create Node to do the convulotional operation with one padding and stride
class Conv(Node):
    def __init__(self, param, var,):
        Node.__init__(self, [*param, var])

    def  forward(self):
        ''' do forward convulutional pass, where a is wieght matrix, b is bias, and x is input '''
        # get input and weights and biases
        b, a, x = self.inputs

        # get shapes
        n_batch, height_in, width_in, in_channels = x.value.shape     
        _, height_out, width_out, _ = x.value.shape
        height_kernel, width_kernel, _, n_kernel = a.value.shape

        # calculating pad size
        pad = tuple(((height_kernel - 1) // 2, (width_kernel - 1) // 2),)

        # padding (code from slides)
        x_pad = np.zeros((n_batch, height_in + 2 * pad[0], width_in+ 2 * pad[1], in_channels),)
        x_pad[:,pad[0]:height_in+pad[0],pad[1]:width_in+pad[1],:] = x.value

        # start the actual operatioon, note that the number of kernels dictates the amount of self.value channels
        self.value = np.zeros((n_batch,height_out,width_in,n_kernel))

        for i in range(height_out):
            for j in range(width_out):
                h_start = i
                h_end = h_start + height_kernel
                w_start = j
                w_end = w_start + width_kernel

                self.value[:, i, j, :] = np.sum(
                    x_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    a.value[np.newaxis, :, :, :],
                    axis=(1, 2, 3)
                )

        self.value = self.value + b.value
    

    def backward(self):
        '''backward pass, self.outputs[0].gradients[self] is the derivative of of the loss to this layer.'''

        b, a, x = self.inputs
        
        # get shape for the gradient of input x
        _, height_out, width_out, _ = self.outputs[0].gradients[self].shape
        n, height_in, width_in, _ = x.value.shape
        height_kernel, width_kernel, _, n_kernel = a.value.shape

        # calculating pad size
        pad = tuple((height_kernel - 1) // 2, (width_kernel - 1) // 2)

        # padding (code from slides)
        x_pad = np.zeros(height_in + 2 * pad[0], width_in+ 2 * pad[1])
        x_pad[pad[0]:height_in+pad[0],pad[1]:width_in+pad[1]] = x.value
        self.value = np.zeros_like(x_pad)

        self.gradients[b] = self.outputs[0].gradients[self].sum(axis=(0, 1, 2)) / n
        self.gradients[a] = np.zeros_like(a.value)

        for i in range(height_out):
            for j in range(width_out):
                h_start = i
                h_end = h_start + height_kernel
                w_start = j
                w_end = w_start + width_kernel
                self.value[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    a.value[np.newaxis, :, :, :, :] *
                    self.outputs[0].gradients[self][:, i:i+1, j:j+1, np.newaxis, :],
                    axis=4
                )
                self.gradients[a] += np.sum(
                    x_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    self.outputs[0].gradients[self][:, i:i+1, j:j+1, np.newaxis, :],
                    axis=0
                )

        self.gradients[a] /= n

        self.gradients[x] = self.value[:, pad[0]:pad[0]+height_in, pad[1]:pad[1]+width_in, :]

class MaxPooling(Node):
    '''Performs 2x2 MaxPooling, reducing the map's width and height by half for each layer'''
    def __init__(self, node=None):
        Node.__init__(self, [node])
    
    def forward(self):
        x = self.inputs[0].value
        
        # get shapes
        n, height_in, width_in, n_channels = x.shape
        height_out = 1 + (height_in - 2) // 2
        width_out = 1 + (width_in - 2) // 2

        self.value = np.zeros((n, height_out, width_out, n_channels))

        for i in range(height_out):
            for j in range(width_out):
                h_start = i << 1
                h_end = h_start + 2
                w_start = j << 1
                w_end = w_start + 2
                x_slice = x[:, h_start:h_end, w_start:w_end, :]
                self.value[:, i, j, :] = np.max(x_slice, axis=(1, 2))

    def backward(self):
        x = self.inputs[0]

        self.gradients[x] = np.zeros_like(self._a)
        n, height_out, width_out, c = self.outputs[0].gradients[self].shape

        for i in range(height_out):
            for j in range(width_out):
                h_start = i << 1
                h_end = h_start + 2
                w_start = j << 1
                w_end = w_start + 2
                self.gradients[x][:, h_start:h_end, w_start:w_end, :] += \
                    self.outputs[0].gradients[self][:, i:i + 1, j:j + 1, :] * np.array([1,0,0,0]*n*c).reshape(3,4,3)