import random
import math
from graphviz import Digraph

# Implements Neurons and Synapses

# the Neuron class
class Neuron:
    def __init__(self, id: int, bias: float = None, activation_f = None):
        # if bias is not passed in, choose at random between 0 - 1
        if bias is None:
            self.bias = random.random()
        else:
            self.bias = bias # the bias of the neuron

        if activation_f is None:
            self.activation_f = 'sigmoid'
        else:
            self.activation_f = activation_f

        self.id: int = id # innovation id of the neuron
        
        self.out_synapses: list[Synapse] = [] # set of synapses that are connected

        self.value: float = 0 # attribute to hold input value to neuron

    def __repr__(self):
        return f'[Neuron {self.id}] Bias: {self.bias}'
    
    def connect(self, synapse: 'Synapse'):
        self.out_synapses.append(synapse)

    def get_synapses(self):
        return [str(synapse) for synapse in self.out_synapses]
    
    def flush_value(self):
        self.value = 0

    def forward(self, f_type = 'sigmoid'):
        # set self.value to activation value
        acvitation_val = self.activation(self.value + self.bias, f_type)
        # then, forward that activation to connected neurons
        for synapse in self.out_synapses:
            synapse.into.value += acvitation_val * synapse.weight

        # consider calling self.flush_value after this to prevent "double forwarding"
    
    def activation(self, raw_activation, f_type):
        if f_type == 'sigmoid':
            return 1 / (1 + math.exp(-1 * raw_activation))
        elif f_type == 'softmax': # this has to implemented on the layer level, as it requires summing other activations
            pass
        elif f_type == 'tanh':
            pos_exp = math.exp(raw_activation)
            neg_exp = math.exp(-1 * raw_activation)
            return (pos_exp - neg_exp) / (pos_exp + neg_exp)
        else:
            raise Exception(f'function type {f_type} is not been implemented')
        
    def get_activation(self):
        return self.value


# the Synapse class
class Synapse:
    def __init__(self, id: int, outof: Neuron, into: Neuron, weight: float, is_on: bool):
        self.id = id # innovation id of the synapse
        self.outof = outof # the origin neuron of the synapse
        self.into = into # the destination neuron of the synapse
        self.weight = weight
        self.is_on = is_on # if the synapse is disabled or not

        # connect this synapses to the neurons its leaving
        outof.connect(self)

    def __repr__(self):
        return f'[Synapse {self.id}] Neuron {self.outof.id} -> Neuron {self.into.id}, Weight: {self.weight}, {"Enabled" if self.is_on else "Disabled"}'
    
    def __str__(self):
        return f'Synapse {self.id}'
    
# Layer will be used for the input and output layer
class Layer:
    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons
        self.n_neurons = len(neurons)

    def add_neuron(self, neuron: Neuron):
        if neuron in self.neurons:
            raise Exception(f'{neuron} is already in this layer')
        else:
            self.neurons.append(neuron)
            self.n_neurons += 1

    def remove_neuron(self, neuron: Neuron):
        if neuron not in self.neurons:
            raise Exception(f'{neuron} is not in this layer')
        else:
            self.neurons.remove(neuron)
            self.n_neurons -= 1

    def forward(self):
        # forward activation amount
        for neuron in self.neurons:
            # forward activation of neuron to next neuron
            neuron.forward()

    def get_activations(self): # utility function to get activation of all neurons in layer
        return [neuron.value for neuron in self.neurons]
            
    def softmax(self): # perform a softmax operation of this layer
        bottom = sum(math.exp(neuron.get_activation()) for neuron in self.neurons)
        for neuron in self.neurons:
            neuron.value = math.exp(neuron.value) / bottom




def visualize_neural_network(neurons):
    dot = Digraph(comment='Neural Network')
    dot.attr(rankdir='LR')
    
    for neuron in neurons:
        dot.node(f'Neuron_{neuron.id}', label=f'Neuron {neuron.id}\n Bias: {neuron.bias:.2f}', shape = 'ellipse', height = '0.5', width = '1', fontsize = '10')

        for synapse in neuron.out_synapses:
            dot.node(f'Synapse_{synapse.id}', label=f'Synapse {synapse.id}\nWeight: {synapse.weight:.2f}', shape = 'rarrow', fontsize = '10')
            dot.edge(f'Neuron_{neuron.id}', f'Synapse_{synapse.id}')
            dot.edge(f'Synapse_{synapse.id}', f'Neuron_{synapse.into.id}')

    return dot