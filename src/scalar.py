# We create our own Scalar structure that keeps memory of which
# other parameters were used for it's initialization (Directed Acyclic Graph) 
# for the purpose of backpropagation

from numpy import log
import sys
#sys.setrecursionlimit(10000)

class Scalar:
    def __init__(self, data : float) -> None:
        self.value = data # the value the node is keeping (float)
        self.edges = [] # points to the nodes that were combined to create this one (ex. z = x + y)
        self.grad = 0 # gradient
        self._backward = None # a function that will propagate the gradient backward

    def __add__(self, other):
        '''the addition operation (z = x + y)'''
        
        if not isinstance(other, Scalar): # make sure other is scalar
            other = Scalar(other)
        result = Scalar(other.value + self.value) # create result scalar
        result.edges.append(self) # add origin node
        result.edges.append(other)# add origin node

        def _backward():
            '''dz / dx = 1
            and dz / dy = 1
            then dL / dx = sum_z(dL / dz * dz/dx) for all z which point to x
            so for this current z, we do dL/dx += dL / dz * dz / dx because of chain rule
            this can be simplified to dL/dx += dL/dz and same for y'''
            self.grad += result.grad
            other.grad += result.grad
        result._backward = _backward
        return result
    
    def __mul__(self, other):
        ''' the multiplication operation z = x * y, only backward differs'''
        if not isinstance(other, Scalar):
            other = Scalar(other)
        result = Scalar(other.value * self.value)
        result.edges.append(self)
        result.edges.append(other)

        def _backward():
            '''dz/dx = y
            dL/dx += dL/dz * y
            same for y (+= dL/dz * x)'''
            self.grad += result.grad * other.value
            other.grad += result.grad * self.value
        
        result._backward = _backward
        return result
    

    def backward(self):
        '''the backpropagation process if we start from this node
        then this node has gradient 1, i.e. dL/dL = 1
        since we work with DAG, and we need to process the nodes in topological order
        to make sure all the gradients flow in order, we use dfs to compute the reverse topological order
        and then find the topological order'''
        topo_order = []
        visited = set() 
        def dfs(u):
            '''dfs that adds the indices in topological order'''
            if u in visited:
                return
            visited.add(u)
            for edge in u.edges:
                dfs(edge)
            topo_order.append(u)

        dfs(self)
        topo_order = reversed(topo_order)

        self.grad = 1 # gradient to iteslf is 1
        for node in topo_order:
            if (node._backward != None):
                # propagate gradients backward
                node._backward()

    def __neg__(self): # -x
        return -1 * self

    def __sub__(self, other): # x - y
        return self + (-other)

    def __truediv__(self, other): # x / y
        if not isinstance(other, Scalar):
            other = Scalar(other)
        return self * (other ** (-1))
    
    def __repr__(self): # printing
        return str(self.value)
    
    def __radd__(self, other): # other + self = self + other
        return self + other
    
    def __rsub__(self, other): # other - self = other + (-self)
        return other + (-self)

    def __rmul__(self, other): # other * self = self * other
        return self * other

    def __rtruediv__(self, other): # other / self = other * self ** -1
        if not isinstance(other, Scalar):
            other = Scalar(other)
        return other * (self ** -1)
    
    def __pow__(self, other):
        '''a little bit unsafe but okay for the simple exponentiations we will use
            issue are negative bases'''
        if not isinstance(other, Scalar):
            other = Scalar(other)

        result = Scalar(self.value ** other.value)
        result.edges = [self, other]

        def _backward():
            # z = x ** y
            # dz/dx = y * x ** (y-1)
            # dz/dy = (x ** y) * ln (x)
            self.grad += result.grad * other.value * (self.value ** (other.value - 1)) #(shouldn't be that high if proper normalizations)
            if self.value > 0: # very unsafe but this will do
                other.grad += result.grad * result.value * log(self.value)

        result._backward = _backward
        return result
    
    def ReLU(self):
        '''standard relu'''
        result = Scalar(max(self.value, 0))
        result.edges.append(self)
        def _backward():
            if self.value > 0:
                self.grad += result.grad
        result._backward = _backward
        return result
    
    def ReLU(X):
        self = X
        '''standard relu as a function'''
        result = Scalar(max(self.value, 0))
        result.edges.append(self)
        def _backward():
            if self.value > 0:
                self.grad += result.grad
        result._backward = _backward
        return result
    
    def clip_grad(self, xmin, xmax):
        if self.grad < xmin:
            self.grad = xmin
        if self.grad > xmax:
            self.grad = xmax
        
