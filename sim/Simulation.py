import Sandbox
import neural_net.nn as nn
import random
from constants import POP_SIZE, N_GENS, SPECIATION_THRESHOLD

class Simulation:
    def create_dense_network(input_count=16, output_count=4):
        """
        helper function to kickstart initial population. Creates a neural network with 16 input and 4 output.
        The input is densely connected to the output. Each of the synapse weights and neuron biases are randomly chosen.
        """
        # Create input and output neurons
        input_neurons = [nn.NeuronGene(i, random.random()) for i in range(0, input_count)]
        output_neurons = [nn.NeuronGene(j + input_count, random.uniform(-1, 1)) for j in range(output_count)]
        
        # Create all possible synapse connections between input and output neurons
        synapse_gene = []
        SIN = 0
        for output_neuron in output_neurons:
            for input_neuron in input_neurons:
                synapse_gene.append(nn.SynapseGene(SIN, input_neuron, output_neuron, random.uniform(-1, 1), True))
                SIN += 1
                
        # Create the NetworkGenome
        network_genome = nn.NetworkGenome(input_neurons, output_neurons, [], synapse_gene)
        nn.NetworkGenome.SIN = SIN
        nn.NetworkGenome.NIN = input_count + output_count
        
        return network_genome

    def __init__(self, population: int = POP_SIZE, generations: int = N_GENS, spec_thres: float = SPECIATION_THRESHOLD):
        self.species = {}