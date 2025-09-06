from engine import Value
import random

class Neuron:
    #nin: number of inputs a neuron should expect
    def __init__(self, nin:int=1):
        self.weights = [Value(data=random.uniform(-1,1), label='weight') for _ in range(nin)]
        self.bias = Value(data=random.uniform(-1,1), label='bias')

    def __call__(self, vals):
        act = sum((weight*val for weight, val in zip(self.weights, vals)), self.bias)
        out = act.tan()
        return out

    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    # nin: number of inputs a layer should handle
    # nout: number of outputs given by layer or number of neurons
    def __init__(self, nin: int, nout: int):
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
    # nin: number of inputs expected for MLP
    # nouts: list of counts of neurons each layer going to have
    def __init__(self, nin: int, nouts: list[int]):
        sn = [nin] + nouts
        self.layers = [Layer(sn[i], sn[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: list[float]) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]