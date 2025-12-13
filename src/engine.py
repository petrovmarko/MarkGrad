from atom import Scalar
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

    def __init__(self):
        self.layers = []
        return
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X if len(X) > 1 else X[0]
    
    def __call__(self, X):
        return self.forward(X)
    
