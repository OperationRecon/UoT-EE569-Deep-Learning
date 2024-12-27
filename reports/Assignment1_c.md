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

## 5. Observations from `full_MNIST_evaluations.txt`

The accuracy of the model varies with different hyperparameters such as depth, width, learning rate, batch size, and epochs. Here are some observations:

- **Depth and Width**:
  - Increasing the depth generally improves accuracy. For example, with `Depth: 3` and `Width: 64`, the accuracy reaches up to 99.50%.
  - Lower depths tend to have lower accuracy. For example, with `Depth: 1` and `Width: 64`, the accuracy is around 83.45% to 97.65%.

- **Learning Rate**:
  - A moderate learning rate (e.g., 0.04) tends to yield better accuracy. For example, with `Depth: 1`, `Width: 64`, and `Learning rate: 0.04`, the accuracy is 93.13%.
  - Very high learning rates (e.g., 0.16) result in lower accuracy. For example, with `Depth: 1`, `Width: 64`, and `Learning rate: 0.16`, the accuracy is 88.48%.

- **Batch Size**:
  - Larger batch sizes (e.g., 300) with moderate learning rates tend to perform well. For example, with `Depth: 1`, `Width: 64`, `Learning rate: 0.04`, and `Batch size: 300`, the accuracy is 92.72%.
  - Smaller batch sizes (e.g., 100) with the same learning rate and depth can also perform well but may take longer to train.

- **Epochs**:
  - More epochs generally improve accuracy. For example, with `Depth: 3`, `Width: 64`, `Learning rate: 0.02`, `Batch size: 100`, and `Epochs: 1000`, the accuracy is 99.50%.
  - Fewer epochs result in lower accuracy. For example, with `Depth: 1`, `Width: 64`, `Learning rate: 0.08`, `Batch size: 150`, and `Epochs: 10`, the accuracy is 83.45%.

In summary, the accuracy improves with increased depth, moderate learning rates, larger batch sizes, and more epochs.

## Summary

The provided files implement a Multi-Layer Perceptron (MLP) to learn the MNIST dataset. The key components include:

- **Core Nodes and Layers**: Defined in `EDF_Percpetron.py` and `Nueron_Layer.py`, these files provide the building blocks for the MLP.
- **MLP Class**: Defined in `MLP.py`, this class combines all MLP functionality and provides methods for training and evaluation.
- **Training Script**: Defined in `full_MNIST.py`, this script loads the dataset, preprocesses the data, defines the MLP architecture, trains the model, and evaluates its performance.

By following the structure and methods provided in these files, the MLP can be effectively trained to classify handwritten digits from the MNIST dataset.