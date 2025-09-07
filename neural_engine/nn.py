from .engine import Value
import random

class Neuron:
    #nin: number of inputs a neuron should expect
    def __init__(self, nin:int=1, act_type='tanh'):
        self.weights = [Value(data=random.uniform(-1,1), label='weight') for _ in range(nin)]
        self.bias = Value(data=random.uniform(-1,1), label='bias')
        self.act_type = act_type

    def __call__(self, vals):
        act = sum((weight*val for weight, val in zip(self.weights, vals)), self.bias)
        if self.act_type == 'tanh':
           return act.tan()
        elif self.act_type == 'sigmoid':
            return act.sig()
        else:
            return act.relu()

    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    # nin: number of inputs a layer should handle
    # nout: number of outputs given by layer or number of neurons
    def __init__(self, nin: int, nout: int, act_type:str="tanh", name=''):
        self.name=act_type+'-'+name
        self.act_type=act_type
        if act_type == 'tanh':
            self.neurons = [Neuron(nin) for _ in range(nout)]
        elif act_type == 'sigmoid':
            self.neurons = [Neuron(nin) for _ in range(nout)]
        else:
            self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: list[float]) -> list[Value]:
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        output = []
        for neuron in self.neurons:
            output.extend(neuron.parameters())
        return output


class MLP:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def __call__(self, x: list[float]) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def mse_loss(exp_out:list[float], res_out: list[Value])-> Value:
    loss = sum( (eo-ro)**2 for eo, ro in zip(exp_out, res_out))
    return loss

