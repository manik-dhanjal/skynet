import math
import graphviz
from graphviz import nohtml

class Value:
    def __init__(self, data: float, _children=(), _op:str=None, label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._children = _children
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value({self.data}, {self.label})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(data = self.data + other.data, _children = (self, other), _op = '+')
        def _backward ():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self+ (-other)

    def __neg__(self):
        return  self * -1

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(data = self.data * other.data, _children = (self, other), _op = '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def tan(self):
        tan_val = (math.exp(2 * self.data) - 1)/(math.exp(2 * self.data) +1)
        out = Value(data =  tan_val, _children=( self, ), _op='tan')
        def _backward():
            self.grad += 1 - tan_val**2
        out._backward = _backward
        return out

    def exp(self):
        exp_val = math.exp(self.data)
        out = Value(exp_val, _op='exp', _children=(self,) )
        def _backward():
            self.grad += exp_val*out.grad
        out._backward = _backward
        return out

    def __pow__(self, power, modulo=None):
        power = power if isinstance(power, Value) else Value(data=power)
        out = Value(self.data**power.data, _op='**', _children=(self, power))
        def _backward():
            self.grad = (power.data* (self.data**(power.data-1)))*out.grad
            power.grad = (math.log(self.data)*(self.data**power.data))*out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(data=other)
        return self * (other**-1)

    def backward(self):
        self.grad = 1
        stack = [self]
        visited = set()
        while len(stack):
            cur = stack.pop()
            if cur not in visited:
                visited.add(cur)
                if cur._backward is not None:
                    cur._backward()
                if cur._children is not None:
                    stack+= cur._children


def draw_graph(parent: Value):
    f = graphviz.Digraph(node_attr={'shape': 'record'}, format='svg', graph_attr={'rankdir': 'LR'})
    stack = [parent];
    f.node(str(id(parent)),nohtml(f'<f0> {parent.label} |<f1> Data: {parent.data}|<f2>Grad: {parent.grad}'))
    while len(stack)!=0:
        cur = stack.pop()
        cur_uid = str(id(cur))
        if cur._op != None :
            f.node(name=cur_uid+cur._op, label=cur._op,shape='oval')
            f.edge(cur_uid+cur._op, cur_uid)
        if cur._children is not None:
            for children in cur._children:
                child_uid = str(id(children))
                f.node(child_uid ,nohtml(f'<f0> {children.label} |<f1> Data: {children.data}|<f2>Grad: {children.grad}'))
                if cur._op != None :
                    f.edge(child_uid, cur_uid+cur._op)
                stack.append(children)
    return f
