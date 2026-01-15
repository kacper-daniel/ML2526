import numpy as np
from typing import List

class DenseLayer():
    def __init__(self, input_size: int, output_size: int, activation: str): 
        '''
            Dense layer

            input_size (int): input vector length
            output_size (int): output vector length
            activation (string): activation function
        '''

        # weightsa and biases initialization
        # he-initialization technique
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

        self.activation = activation

        # variables for adam optimizer
        self.m_weights = np.zeros((input_size, output_size))
        self.v_weights = np.zeros((input_size, output_size))
        self.m_biases = np.zeros((1, output_size))
        self.v_biases = np.zeros((1, output_size))
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def forward(self, x):
        '''
            Forward pass

            x (Tensor): Data
        '''
        self.x = x

        # compute output
        z = np.dot(self.x, self.weights) + self.biases

        if self.activation == "relu":
            self.output = np.maximum(0, z)
        elif self.activation == "softmax":
            exp_values = np.exp(z - np.max(z, axis=-1, keepdims=True)) 
            self.output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        else:
            print(f"Activation function does not exist: '{self.activation}' ")

        return self.output
    
    def backward(self, d_values: List[float], lr: float, t: int):
        '''
            Backward pass with Adam optimization

            d_values (List[float]): output derivatives
            lr (float): learning rate constant
            t (int): timestep
        '''
        # softmax derivative
        if self.activation == "softmax":
            k = np.sum(d_values * self.output, axis=1, keepdims=True)
            d_values = self.output * (d_values - k)

        # relu derivative
        elif self.activation == "relu":
            d_values = d_values * (self.output > 0)

        # calculate derivatives with respect to the weight and bias 
        d_weights = np.dot(self.x.T, d_values)
        d_biases = np.sum(d_values, axis=0, keepdims=True)
        # crop derivatives to avoid too big or too small numbers
        d_weights = np.clip(d_weights, -1.0, 1.0)
        d_biases = np.clip(d_biases, -1.0, 1.0)

        # calculate gradient with respect to input
        d_inputs = np.dot(d_values, self.weights.T)

        # update weights
        self.weights -= lr * d_weights
        self.biases -= lr * d_biases

        # update wieghts using m and v
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * d_weights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (d_weights ** 2)
        m_hat_weights = self.m_weights / (1 - self.beta1 ** t)
        v_hat_weights = self.v_weights / (1 - self.beta2 ** t)
        self.weights -= lr * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

        # updates biases using m and v 
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * d_biases
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (d_biases ** 2)
        m_hat_biases = self.m_biases / (1 - self.beta1 ** t)
        v_hat_biases = self.v_biases / (1 - self.beta2 ** t)
        self.biases -= lr * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

        return d_inputs