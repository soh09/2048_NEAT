import random
import math
from graphviz import Digraph

# function to visualize neural network
def visualize_neural_network(neurons):
    dot = Digraph(comment='Neural Network', format='png')
    dot.attr(rankdir='LR')  # Set the rank direction to left to right

    for neuron in neurons:
        label = f'#{neuron.id} | Bias={neuron.bias:.2f}\nval={neuron.value:.4f}'
        dot.node(f'Neuron_{neuron.id}', label=label, shape='ellipse', width='0.1', height='0.5', fontsize='10')

        for synapse in neuron.out_synapses:
            label = f'w={synapse.weight:.2f}'
            dot.edge(f'Neuron_{neuron.id}', f'Neuron_{synapse.into.id}', label=label, fontsize = '10')

    return dot

# Implements Neurons, Synapses, Layer, and Network

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


class Network:
    def __init__(self, input_layer: Layer, output_layer: Layer) -> None:
        self.input_l = input_layer
        self.output_l = output_layer

        self.fitness = 0 # used when there's conflicting genes during crossover
        self.neurons = self.input_l.neurons + self.output_l.neurons
        self.sorted_neurons = []

    def set_input(self, inputs): # sets the input neurons' values
        if len(inputs) != self.input_l.n_neurons:
            raise Exception(f'Input size mismatch: Expected {self.input_l.n_neurons}, got {len(inputs)}')
        else:
            for neuron, val in zip(self.input_l.neurons, inputs):
                neuron.value = val

    def forward(self):
        visited = set()
        sorted_neurons = []

        def dfs(visited_set: set, top_sort: list[Neuron], neuron: Neuron): # depth-first search method as helper
            for synapse in neuron.out_synapses:
                connected = synapse.into
                if connected not in visited_set:
                    dfs(visited_set, top_sort, connected)
            visited_set.add(neuron)
            sorted_neurons.append(neuron)


        # do topological sort of synapses, and call forward
        for neuron in self.neurons: # while all neurons haven't been discovered
            if neuron not in visited:
                dfs(visited, sorted_neurons, neuron)

        self.sorted_neurons = list(reversed(sorted_neurons)) # topologically sorted neurons

        for neuron in self.sorted_neurons:
            neuron.forward()
    
    def add_hidden_neuron(self, neuron: Neuron):
        self.neurons.append(neuron)

    def __repr__(self):
        return f'Neural Network with {len(self.neurons)} neurons'