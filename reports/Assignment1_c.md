# Report: Using an MLP to Learn the MNIST Dataset

This report details the key parts of the provided files for using a Multi-Layer Perceptron (MLP) to learn the MNIST dataset. The files include the implementation of the MLP, the necessary layers and nodes, and the script to train and evaluate the model.

## 1. File: `EDF_Percpetron.py`

This file contains the core components of the neural network, including the base `Node` class, various types of nodes (e.g., `Input`, `Parameter`, `Linear`, `ReLU`, `Softmax`, `Cross_Entropy`), and the `Conv` and `MaxPooling` layers.

- **Node Class**: The base class for all nodes in the network, providing the structure for forward and backward passes.
- **Input Node**: Represents the input data.
- **Parameter Node**: Represents trainable parameters (weights and biases).
- **Linear Node**: Implements a linear transformation (fully connected layer).
- **ReLU Node**: Implements the ReLU activation function.
- **Softmax Node**: Implements the softmax activation function for classification.
- **Cross_Entropy Node**: Implements the cross-entropy loss function for multi-class classification.
- **Conv and MaxPooling Nodes**: Implement convolutional and max-pooling operations for CNNs.

## 2. File: `Nueron_Layer.py`

This file defines the layers of the neural network, including input layers, computation layers, and specific types of computation layers (e.g., `Linear_Computation_Layer`, `Linear_Softmax_Computation_Layer`, `Conv_layer`).

- **Nueron_Layer Class**: The base class for all layers, providing the structure for forward and backward passes.
- **Input_Layer Class**: Represents the input layer of the network.
- **Computation_Layer Class**: Represents a layer with an operation and activation function, as well as trainable parameters.
- **Linear_Computation_Layer Class**: A computation layer with a linear transformation and an activation function.
- **Linear_Softmax_Computation_Layer Class**: A computation layer with a linear transformation followed by a softmax activation function.
- **Conv_layer Class**: A convolutional layer with optional max-pooling.

## 3. File: `MLP.py`

This file defines the `MLP` class, which combines all MLP functionality into one class to reduce code clutter. It includes methods for building the network, training, and evaluation.

- **MLP Class**:
  - **Initialization**: Initializes the MLP with the specified number of features, outputs, depth, width, hidden layer type, output layer type, loss node type, and test node.
  - **Learn Method**: Trains the MLP on the training data for a specified number of epochs, batch size, and learning rate.
  - **Evaluate Method**: Evaluates the MLP on the test data and calculates accuracy and average entropy.
  - **Forward Pass Method**: Performs a forward pass through the network.
  - **Backward Pass Method**: Performs a backward pass through the network.
  - **SGD Update Method**: Updates the trainable parameters using stochastic gradient descent.

## 4. File: `full_MNIST.py`

This file contains the script to load the MNIST dataset, preprocess the data, define the MLP architecture, train the model, and evaluate its performance.

- **Loading the Dataset**: Loads the MNIST dataset using Keras and reshapes the data to fit the MLP model.
- **Defining Constants**: Defines constants such as the number of features, number of outputs, learning rate, number of epochs, batch size, depth, and width of the MLP.
- **Evaluation Function**: Defines a function to evaluate the model's predictions against the true labels.
- **Hot-One Encoding**: Converts the labels to one-hot encoded vectors.
- **MLP Initialization**: Initializes the MLP with the specified architecture.
- **Training the MLP**: Trains the MLP on the training data and logs the loss and learning time.
- **Evaluating the MLP**: Evaluates the MLP on the test data and logs the accuracy and average entropy.

## Summary

The provided files implement a Multi-Layer Perceptron (MLP) to learn the MNIST dataset. The key components include:

- **Core Nodes and Layers**: Defined in `EDF_Percpetron.py` and `Nueron_Layer.py`, these files provide the building blocks for the MLP.
- **MLP Class**: Defined in `MLP.py`, this class combines all MLP functionality and provides methods for training and evaluation.
- **Training Script**: Defined in `full_MNIST.py`, this script loads the dataset, preprocesses the data, defines the MLP architecture, trains the model, and evaluates its performance.

By following the structure and methods provided in these files, the MLP can be effectively trained to classify handwritten digits from the MNIST dataset.