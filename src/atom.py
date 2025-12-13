# We create our own Scalar structure that keeps memory of which
# other parameters were used for it's initialization (Directed Acyclic Graph) 
# for the purpose of backpropagation

from cmath import log

class Scalar:
    def __init__(self, data) -> None:
        self.value = data
        self.edges = []
        self.grad = 0
        self._backward = None

    def __add__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        result = Scalar(other.value + self.value)
        result.edges.append(self)
        result.edges.append(other)

        def _backward():
            self.grad += result.grad
            other.grad += result.grad
        result._backward = _backward
        return result
    
    def __mul__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        result = Scalar(other.value * self.value)
        result.edges.append(self)
        result.edges.append(other)

        def _backward():
            self.grad += result.grad * other.value
            other.grad += result.grad * self.value
        
        result._backward = _backward
        return result
    

    def backward(self):
        
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
                node._backward()

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, other):
        # z = x ^ y = e ^ (ln(x)) ^ y = e ^ (ln(x) * y)
        # dz/dx = y*x^(y-1)
        # dz/dy = e ^ (ln(x) * y) * ln(x) = x ^ y * ln(x)
        if not isinstance(other, Scalar):
            other = Scalar(other)
        result = Scalar(self.value ** other.value)
        result.edges.append(self)
        result.edges.append(other)

        def _backward():
            self.grad += result.grad * other.value * (self.value ** (other.value-1))
            other.grad += result.grad * (result.value) * log(self.value)
        result._backward = _backward

        return result

    def __truediv__(self, other):
        return self * (other ** (-1))
    
    def __repr__(self):
        return str(self.value)
    
    def __radd__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return (self ** -1) * other
    
    def ReLU(self):
        self.value = max(self.value, 0)