# Neural Network Engine (MLP)
# currently it does not support paralel computation (whole mini_batch at once)
from scalar import Scalar
import random

class Parameter:
    def __init__(self, in_features, activation):
        '''contains w_i for each in feature and one bias
        the activation is a seperate class'''
        self.w = [Scalar(random.random()) for x in range(in_features)]
        self.b = Scalar(random.random())
        self.in_features = in_features
        if activation == 'relu':
            self.activation = Scalar.ReLU
        else:
            self.activation = lambda x : x
    
    def forward(self, X):
        # Make sure it's in the form of array
        if isinstance(X, (float, int, Scalar)):
            X = [X]
        return self.activation(sum([X[i] * self.w[i] for i in range(self.in_features)]) + self.b)
    
    def __call__(self, X):
        return self.forward(X)
    
    def __mul__(self, y):
        self.w = [w * y for w in self.w ]
        self.b = self.b * y
        return self
    
class Layer:

    def __init__(self, in_feature, out_features, scale = 10, activation = True):
        self.layer = [Parameter(in_feature, activation) * scale for i in range(out_features)]

    def forward(self, X):
        return [parameter(X) for parameter in self.layer]
    
    def __call__(self, X):
        return self.forward(X)

class NeuralNet:
    def __init__(self, lr):
        '''only learning rate as parameter'''
        self.layers = []
        self.parameters = []
        self.lr = lr
        return
    
    def add_layer(self, layer):
        '''Add a layer to the network'''
        self.layers.append(layer)
        for neuron in layer.layer:
            self.parameters.append(neuron)
    
    def forward(self, X):
        '''process the layers one by one and substituting X as the result'''
        # X -> layer1(x) -> layer2(layer1(x)) -> ...
        for layer in self.layers:
            X = layer(X)

        # If X is length one, just return the only element
        return X if len(X) > 1 else X[0]
    
    def __call__(self, X):
        return self.forward(X)
    
    def zero_grad(self):
        '''Zero out the gradients of all parameters'''
        for param in self.parameters:
            for neuron in param.w:
                neuron.grad = 0
                neuron.edges = []
            param.b.grad = 0
            param.b.edges = []
            

    def step(self):
        '''Update the values of the parameters'''
        for param in self.parameters:
            for neuron in param.w:
                neuron.clip_grad(-10,10) # clip gradients to not explode
                neuron.value -= self.lr * neuron.grad
            param.b.clip_grad(-10,10) # clip gradients to not explode
            param.b.value -= self.lr * param.b.grad
