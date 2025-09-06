import math

class Value:
    def __init__(self, data, _children=(), _op=None, label=''):
        self.data = data
        self._children = _children
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value({self.data}, {self.label})"

    def __add__(self, other):
        out = Value(data = self.data + other.data, _children = (self, other), _op = '+')
        return out

    def __mul__(self, other):
        out = Value(data = self.data * other.data, _children = (self, other), _op = '*')
        return out

    def __truediv__(self, other):
        out = Value(data = self.data / other.data, _children=(self, other), _op = '/')
        return out

    def tan(self):
        tan_val = (math.exp(self.data ** 2) - 1)/(math.exp(self.data**2) +1)
        out = Value(data =  tan_val, _children=( self ), _op='tan')
        return out

