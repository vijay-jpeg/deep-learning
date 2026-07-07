
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
        if isinstance(other, Value):
            new_data = self.data + other.data
        else: 
            new_data = self.data + other
            other = Value(other)
        out = self._make(data=new_data, _children=(self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __sub__(self, other):
        if isinstance(other, Value):
            other.data = -other.data
        else: other = -other

        return self.__add__(other)
        
    def __mul__(self, other):
        if isinstance(other, Value):
            new_data = self.data * other.data
        else: 
            new_data = self.data * other
            other = Value(other)
        out = self._make(data=new_data, _children=(self, other), _op='*')

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

    
    def __neg__(self):
        return self.__mul__(-1)
    
    def __truediv__(self, other):
        if isinstance(other, Value):
            other.data = other.data ** -1
        else: 
            other = other ** -1
        return self.__mul__(other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other.__sub__(self)
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other.__truediv__(self)    

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
