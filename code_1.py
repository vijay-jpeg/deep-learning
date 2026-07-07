
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
        return f"Value(data={self.data}, op='{self._op}')"
    
    def __add__(self , other):
        if isinstance(other, Value):
            new_data = self.data + other.data
        else: 
            new_data = self.data + other
            other = Value(other)
        return self._make(data=new_data, _children=(self, other), _op='+')
    
    def __sub__(self , other):
        if isinstance(other, Value):
            new_data = self.data + (-other.data)
        else: 
            new_data = self.data + -(other)
            other = Value(other)
        return self._make(data=new_data, _children=(self, other), _op='-')
    
    def __mul__(self , other):
        if isinstance(other, Value):
            new_data = self.data * other.data
        else: 
            new_data = self.data * other
            other = Value(other)
        return self._make(data=new_data, _children=(self, other), _op='*')
    
    def __pow__(self, exponent: int | float):
        return self._make(data=self.data ** exponent, _children=(self,), _op=f'**{exponent}')
    
    def __neg__(self):
        return self._make(data=self.data * -1, _children=(self,), _op='*-1')
    
    def __truediv__(self, other):
        if isinstance(other, Value):
            new_data = self.data * other.data ** -1
        else: 
            new_data = self.data * other ** -1
            other = Value(other)
        return self._make(data=new_data, _children=(self, other), _op='/')
    
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
    



