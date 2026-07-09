from math import tanh, exp

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None

    def _make(self, data, _children, _op):
        return type(self)(data, _children, _op)
    
    def __repr__(self):
        return f"Value(data={self.data}, op='{self._op}', grad={self.grad})"
    
    def __add__(self, other):
        if not isinstance(other, Value): other = Value(other)
        out = self._make(data=self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
        
    def __mul__(self, other):
        if not isinstance(other, Value): other = Value(other)
        out = self._make(data=self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, exponent):
        if not isinstance(exponent, int | float):
            exponent = float(exponent)
        out = self._make(data=self.data ** exponent, _children=(self,), _op=f'**{exponent}')

        def _backwards():
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad
        out._backward = _backwards

        return out
    
    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other - self
    
    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other / self
    
    def tanh(self):
        out = self._make(data=tanh(self.data), _children=(self,), _op='tanh')

        def _backwards():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backwards

        return out
    
    def exp(self):
        out = self._make(data=exp(self.data), _children=(self,), _op='exp')

        def _backwards():
            self.grad += out.data * out.grad
        out._backward = _backwards

        return out
    
    def relu(self):
        out = self._make(data=0 if self.data < 0 else self.data, _children=(self,), _op='relu')

        def _backwards():
            if out.data > 0: self.grad += out.grad 
        out._backward = _backwards

        return out
    
    def backward(self):
        visited = set()
        topo = []
        def topo_sort(root):
            if root not in visited:
                visited.add(root)

                for child in root._prev:
                    topo_sort(child)

                topo.append(root)

        topo_sort(self)

        self.grad = 1.0

        for v in reversed(topo): v._backward()

def topo_sort(root):
    visited = set()
    order = []

    def build(node):
        if node not in visited:
            visited.add(node)

            for child in node._prev:
                build(child)

            order.append(node)

    build(root)
    return order

def trace(root: Value):
    nodes = list(root._prev)
    nodes.append(root)
    edges = set()
    for prev_node in root._prev: edges.add((prev_node, root))
    
    seen = root._prev.copy()
    for node in nodes:
        for prev_node in node._prev:
            edges.add((prev_node, node))
            if prev_node not in seen:
                nodes.append(prev_node)
                seen.add(prev_node)

    return (set(nodes), edges)


