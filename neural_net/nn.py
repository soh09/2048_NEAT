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


class NeuronGene: # class for holding neuron information, but doesn't have neuron functionality
    def __init__(self, id: int, bias: float = None, activation_f = None):
        self.id = id
        self.bias = bias
        self.activation_f = activation_f

        ################## not sure if this is the right approach
        self.expressed_neuron = self.express(self)
    
    def express(self):
        return Neuron(self.id, self.bias, self.activation_f)

    def __repr__(self):
        return f'[Neuron {self.id}] Bias: {self.bias}'
    
    # crossover only gets called when the id are the same
    @staticmethod
    def crossover(gene1: 'NeuronGene', gene2: 'NeuronGene'):
        if gene1.id != gene2.id:
            raise Exception(f'NeuronGene ids not matching: {gene1.id} != {gene2.id}')
        bias = random.choice([gene1.bias, gene2.bias])
        return NeuronGene(gene1.id, bias, gene1.activation_f)

class SynapseGene: # class for holding neuron information, but doesn't have synapse functionality
    def __init__(self, id: int, outof: NeuronGene, into: NeuronGene, weight: float, is_on: bool): ############## how to find neuron by IDs??
        self.id = id # innovation id of the synapse
        self.outof = outof # the origin neuron id of the synapse
        self.into = into # the destination neuron id of the synapse
        self.weight = weight
        self.is_on = is_on # if the synapse is disabled or not

    def express(self):
        return Synapse(self.id, self.outof.expressed_neuron, self.into.expressed_neuron, self.weight, self.is_on)

    def __repr__(self):
        return f'[Synapse {self.id}] Neuron {self.outof.id} -> Neuron {self.into.id}, Weight: {self.weight}, {"Enabled" if self.is_on else "Disabled"}'
    
    # crossover only gets called when the id are the same
    @staticmethod
    def crossover(gene1: 'SynapseGene', gene2: 'SynapseGene'):
        if gene1.id != gene2.id: # assert that ids are the same 
            raise Exception(f'NeuronGene ids not matching: {gene1.id} != {gene2.id}')
        if gene1.outof.id != gene2.outof.id:
            raise Exception(f'NeuronGene OutOf ids not matching: {gene1.outof.id} != {gene2.outof.id}')
        if gene1.into.id != gene2.into.id:
            raise Exception(f'NeuronGene InTo ids not matching:  {gene1.into.id} != {gene2.into.id}')
        is_on = random.choice([gene1.is_on, gene2.is_on])
        weight = random.choice([gene1.weight, gene2.weight])
        return SynapseGene(gene1.id, gene1.into_id, gene1.outof_id, weight, is_on)

        
# This class holds all the genetic information of a network
# Such as information about deactivated synapses, all neurons, and all synapses
class NetworkGenome:
    def __init__(self, input_neurons: list[NeuronGene], output_neurons: list[NeuronGene], neuron_gene: list[NeuronGene], synapse_gene: list[SynapseGene]):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.neuron_gene = neuron_gene 
        self.synapse_gene = synapse_gene

        self.neuron_ids = set([g.id for g in neuron_gene + input_neurons + output_neurons])
        self.synapse_ids = set([s.id for s in synapse_gene])

    def find_neuron(self, id):
        for ng in self.neuron_gene:
            if ng.id == id:
                return ng
        raise Exception(f'id {id} not found')

    def find_synapse(self, id):
        for sg in self.synapse_gene:
            if sg.id == id:
                return sg
        raise Exception(f'id {id} not found')

    def __repr__(self):
        return f'Neural Network Genome with {len(self.neuron_gene)} neurons, {len(self.synapse_gene)} synapses'
    
    def mutate(self):
        # neuron add/removal
            # ensure input neurons dont get deleted
        # synapse activation/deactivation
        # synapse weight change
        # neuron bias change

        pass # function shouldn't have any return as it modifies internal state
    

    # will create a static method that takes in two NetworkGenes, crossover, and return the offspring
    @staticmethod
    def crossover(parent1: 'Network', parent2: 'Network'):
        # assigns dominant to fitter Network
        dominant, recessive = (parent1, parent2) if parent1.fitness > parent2.fitness else (parent2, parent1)

        neuron_gene = []
        synapse_gene = []

        for ng in dominant.genome.neuron_gene:
            if ng.id in recessive.genome.neuron_ids: 
                # if this NeuronGene is in the recessive NetworkGenome, do crossover
                hybrid_neuron = NeuronGene.crossover(ng, recessive.genome.find_neuron(ng.id))
                neuron_gene.append(hybrid_neuron)
            else:
                # if this neuron is only present in the dominant, add it to the genome
                neuron_gene.append(ng)

        for sg in dominant.genome.synapse_gene:
            if sg.id in recessive.genome.synapse_ids:
                # if this SynapseGene is in the recessive NetworkGenome, do crossover
                hybrid_synapse = SynapseGene.crossover(sg, recessive.genome.find_synapse(sg.id))
                synapse_gene.append(hybrid_synapse)
            else:
                # if this synapse is only present in the dominant, add it to the genome
                synapse_gene.append(sg)

        return NetworkGenome(parent1.input_l, parent2.output_l, neuron_gene, synapse_gene)


class Network:
    #################### change so that initializes from NetworkGenome instance
    def __init__(self, genome: NetworkGenome):
        self.input_l = Layer([n.expressed_neuron for n in genome.input_neurons])
        self.output_l = Layer([n.expressed_neuron for n in genome.output_neurons])
        self.genome = genome

        self.fitness = 0 # used when there's conflicting genes during crossover
        self.neurons = self.input_l.neurons + self.output_l.neurons + [n.expressed_neuron for n in genome.neuron_gene]
        self.sorted_neurons = list[Neuron]

        self.synapses = [s.express() for s in genome.synapse_gene]

        # connects the synapses to the corresponding neurons
        self.connect()

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

    def connect(self): ######################### must implement
        # loop through synapses to connect them to the proper neurons using neuron.connect(synapse)
        pass
    

    # utility functions
    def add_hidden_neuron(self, neuron: Neuron): # unused function, because neurons will not be added to networks, but genome instead
        self.neurons.append(neuron)

    def add_hidden_synapse(self, synapse: Synapse): # likewise to add_hidden_neuron
        self.synapses.append(synapse)

    def __repr__(self):
        return f'Neural Network with {len(self.neurons)} neurons, {len(self.synapses)} synapses'