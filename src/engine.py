# Neural Network Engine (MLP)

from Scalar import Scalar
import random

class Parameter:
    def __init__(self, in_features, activation = True):
        self.w = [Scalar(random.random()) for x in range(in_features)]
        self.b = Scalar(random.random())
        self.in_features = in_features
        self.activation = activation
    
    def forward(self, X):
        if isinstance(X, (float, int)):
            X = [X]
        return sum([X[i] * self.w[i] for i in range(self.in_features)]) + self.b
    
    def __call__(self, X):
        return self.forward(X)
    
class Layer:

    def __init__(self, in_feature, out_features, activation = True):
        self.layer = [Parameter(in_feature, activation) for i in range(out_features)]

    def forward(self, X):
        return [parameter(X) for parameter in self.layer]
    
    def __call__(self, X):
        return self.forward(X)

class NeuralNet:

    def __init__(self, lr):
        self.layers = []
        self.parameters = []
        self.lr = lr
        return
    
    def add_layer(self, layer):
        self.layers.append(layer)
        for neuron in layer.layer:
            self.parameters.append(neuron)
    
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X if len(X) > 1 else X[0]
    
    def __call__(self, X):
        return self.forward(X)
    
    def zero_grad(self):
        for param in self.parameters:
            for neuron in param.w:
                self.grad = 0
            param.b.grad = 0

    def step(self):
        for param in self.parameters:
            for neuron in param.w:
                neuron += -self.lr * neuron.grad
            param.b += -self.lr * param.b.grad


    
