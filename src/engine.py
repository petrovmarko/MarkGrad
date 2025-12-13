from atom import Scalar
import random 
class Parameter:

    def __init__(self, w, b, in_features, activation = True):
        self.w = [Scalar(random.rand()) for x in range(in_features)]
        self.b = Scalar(b)
        self.in_features = in_features
        self.activation = activation
    
    def __call__(self, X):
        assert (len(X) == self.in_features)
        return sum([X[i] * self.w[i] for i in range(X)]) + self.b
    
    
class Layer:

    def __init__(self, layer):
        self.layer = layer
    
    def forward(self, X):
        y = []
        for parameter in self.layer:
            y.append(parameter.forward(X))
        return y

class NeuralNet:

    def __init__(self):
        self.layers = []
        return
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, X):

