import numpy as np
import matplotlib.pyplot as plt

class MultiLayerPerceptron:
    def __init__(self, layer_sizes, activation='sigmoid', learning_rate=0.1):
        """
        Initialize a Multi-Layer Perceptron
        
        Parameters:
        layer_sizes (list): List of integers representing the number of neurons in each layer
                           (including input and output layers)
        activation (str): Activation function to use ('sigmoid' or 'relu')
        learning_rate (float): Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        
        # Set activation function
        if activation == 'sigmoid':
            self.activation = self._sigmoid
            self.activation_derivative = self._sigmoid_derivative
        elif activation == 'relu':
            self.activation = self._relu
            self.activation_derivative = self._relu_derivative
        else:
            raise ValueError("Activation function must be 'sigmoid' or 'relu'")
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(1, self.num_layers):
            # Initialize weights with small random values
            w = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 0.1
            b = np.zeros((1, layer_sizes[i]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """Derivative of ReLU function"""
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        """
        Forward propagation
        
        Parameters:
        X (numpy.ndarray): Input data of shape (n_samples, input_size)
        
        Returns:
        list: Activations of each layer including input
        list: Pre-activation values of each layer
        """
        activations = [X]
        pre_activations = []
        
        A = X
        for i in range(self.num_layers - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            pre_activations.append(Z)
            A = self.activation(Z)
            activations.append(A)
            
        return activations, pre_activations
    
    def backward(self, X, y, activations, pre_activations):
        """
        Backward propagation
        
        Parameters:
        X (numpy.ndarray): Input data
        y (numpy.ndarray): Target values
        activations (list): Activations from forward propagation
        pre_activations (list): Pre-activations from forward propagation
        
        Returns:
        list: Weight gradients
        list: Bias gradients
        """
        m = X.shape[0]
        
        # Calculate output layer error (delta)
        delta = activations[-1] - y
        
        weight_gradients = []
        bias_gradients = []
        
        # Loop backwards through layers
        for layer in range(self.num_layers - 2, -1, -1):
            # Calculate weight and bias gradients
            weight_grad = np.dot(activations[layer].T, delta) / m
            bias_grad = np.sum(delta, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
            
            # Calculate delta for next layer (if not at input layer)
            if layer > 0:
                delta = np.dot(delta, self.weights[layer].T) * self.activation_derivative(activations[layer])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def train(self, X, y, epochs=1000, batch_size=None, verbose=True):
        """
        Train the neural network
        
        Parameters:
        X (numpy.ndarray): Training data
        y (numpy.ndarray): Training targets
        epochs (int): Number of training epochs
        batch_size (int): Size of mini-batches (None for batch gradient descent)
        verbose (bool): Whether to print progress
        
        Returns:
        list: Training loss history
        """
        m = X.shape[0]
        loss_history = []
        
        for epoch in range(epochs):
            # Mini-batch gradient descent
            if batch_size:
                indices = np.random.permutation(m)
                for i in range(0, m, batch_size):
                    batch_indices = indices[i:min(i + batch_size, m)]
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    
                    # Forward pass
                    activations, pre_activations = self.forward(X_batch)
                    
                    # Backward pass
                    weight_gradients, bias_gradients = self.backward(X_batch, y_batch, activations, pre_activations)
                    
                    # Update parameters
                    self.update_parameters(weight_gradients, bias_gradients)
            
            # Batch gradient descent
            else:
                # Forward pass
                activations, pre_activations = self.forward(X)
                
                # Backward pass
                weight_gradients, bias_gradients = self.backward(X, y, activations, pre_activations)
                
                # Update parameters
                self.update_parameters(weight_gradients, bias_gradients)
            
            # Calculate and store loss
            activations, _ = self.forward(X)
            predictions = activations[-1]
            loss = np.mean(np.square(predictions - y))
            loss_history.append(loss)
            
            # Print progress
            if verbose and epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")
        
        return loss_history
    
    def predict(self, X):
        """Make predictions for input data X"""
        activations, _ = self.forward(X)
        return activations[-1]


# Example usage
if __name__ == "__main__":
    # Generate a simple dataset (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train MLP
    mlp = MultiLayerPerceptron(layer_sizes=[2, 4, 1], activation='sigmoid', learning_rate=0.5)
    loss_history = mlp.train(X, y, epochs=5000, verbose=True)
    
    # Make predictions
    predictions = mlp.predict(X)
    print("\nPredictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")
    
    # Plot loss history
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # Plot decision boundary (only works for 2D input)
    if X.shape[1] == 2:
        h = 0.01
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Spectral, edgecolors='k')
        plt.title('Decision Boundary')
        plt.show()
