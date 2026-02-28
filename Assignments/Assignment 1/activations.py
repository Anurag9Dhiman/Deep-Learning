"""
Activation functions and their derivatives for MLP.
"""
import numpy as np 

def sigmoid(z):
    "Numerically stable sigmoid activation."
    # clip to avoid overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    "Derivative of sigmoid: sigma(z) * (1 - sigma(z))."
    s = sigmoid(z)
    return s*(1-s)

def tanh(z):
    "Hyperbolic tangent activation."
    return np.tanh(z)

def tanh_derivative(z):
    "derivative of tan hyperbolic: 1- tanh^2(z)."
    return 1.0 - np.tanh(z)**2
    
def relu(z):
    "Rectified linear unit activation."
    return np.maximum(0,z)

def relu_derivative(z):
    "Derivative of ReLU: 1 if z > 0, else 0"
    return (z>0).astype(float)

def softmax(z):
    "Numerically stable softmax"
    # subtract max for numerical staibility
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Mapping from string name to functions

ACTIVATIONS = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu
}

ACTIVATION_DERIVATIVES = {
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
    'relu': relu_derivative,
}

def get_activation(name):
    "Return activation function by name"
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]

def get_activation_grad(name):
    "Return activation derivative by name."
    if name not in ACTIVATION_DERIVATIVES:
        raise ValueError(f"Unknown activation: {name}.")
    return ACTIVATION_DERIVATIVES[name]

