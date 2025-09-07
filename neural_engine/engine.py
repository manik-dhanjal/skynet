import math

class Value:
    def __init__(self, data: float, _children=(), _op:str=None, label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._children = _children
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, _op={self._op}, label={self.label})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(data = self.data + other.data, _children = (self, other), _op = '+')
        def _backward ():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(data = self.data * other.data, _children = (self, other), _op = '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tan(self):
        x = self.data
        t = (math.exp(2 * x) - 1)/(math.exp(2 * x) +1)
        out = Value(data =  t, _children=( self, ), _op='tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def sig(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(data = s, _children=(self,), _op='sigmoid')
        def _backward():
            return s*(1-s) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        exp_val = math.exp(self.data)
        out = Value(exp_val, _op='exp', _children=(self,) )
        def _backward():
            self.grad += exp_val*out.grad
        out._backward = _backward
        return out


    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()