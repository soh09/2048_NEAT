from Sandbox import Sandbox as Sandbox
import neural_net.nn as nn
import random
from copy import deepcopy
from constants import POP_SIZE, SPECIATION_THRESHOLD, W_DISJOINT, W_EXCESS, W_WEIGHT



class Simulation:
    '''
    Simulation workflow:
    1. Call init
        - initialize 1000 structurally identical neural networks
            - weights are random
        - construct species dictionary
            - since the only differences are weight differences, there will likely only be one species
        - at this point, self.sandboxes is empty
    2. Call mutate_and_speciate
        - this function will mutate the genomes, and update the species dictionary
        - at this point, we are ready to do a simulation run
    3. Call simulate
        - 
    '''
    def __init__(
            self, 
            population: int = POP_SIZE, 
            spec_thres: float = SPECIATION_THRESHOLD, 
            w_disjoint: float = W_DISJOINT, 
            w_excess: float = W_EXCESS, 
            w_weight: float = W_WEIGHT):
        
        self.pop_size = population
        self.speciation_threshold = spec_thres
        self.w_disjoint = w_disjoint
        self.w_excess = w_excess
        self.w_weight = w_weight

        self.species_counter = 0
        
        self.current_gen = 0
        self.species = {}
        self.species_count = {}
        self.genomes: list [nn.NetworkGenome] = []
        self.sandboxes: list [Sandbox] = []

        for i in range(population):
            genome = Simulation.create_dense_network()
            # net = nn.Network(genome)
            # self.sandboxes.append(Sandbox(net))

            # if first genome, that will automatically be the progenitor 
            if i == 0:
                # progenitor has to be deepcopy, because this genome will be modified in-place later when mutated
                self.species[self.species_counter] = {'progenitor': deepcopy(genome), 'children': []}
                self.species_counter += 1
            else:
                # even though the rest should be classified as the same species, well check just in case
                for species_num in range(self.species_counter):
                    dist = nn.NetworkGenome.distance(self.species[species_num]['progenitor'], genome, self.w_disjoint, self.w_excess, self.w_weight)
                    if dist < self.speciation_threshold: # if genome is the same species as species_num
                        self.species[species_num]['children'].append(genome)
                        break
                print('new species')
                self.species[self.species_counter] = {'progenitor': deepcopy(genome), 'children': []}
                self.species_counter += 1

        print('1000 Sandboxes created, ready for simulation')

    def mutate_and_speciate(self):
        # reset self.species
        for species_num in self.species:
            self.species[species_num]['children'] = []
        
        # mutate genomes
        for genome in self.genomes:
            genome.mutate()
            # then put in appropriate species
            for species_num in self.species:
                dist = nn.NetworkGenome.distance(self.species[species_num]['progenitor'], genome, self.w_disjoint, self.w_excess, self.w_weight)
                if dist < self.speciation_threshold: # if genome is the same species as species_num
                    self.species[species_num]['children'].append(genome)
                    break
            # if code reaches here, this genome is a new species
            print('new species')
            self.species[self.species_counter] = {'progenitor': deepcopy(genome), 'children': []}
            self.species_counter += 1

        # remove species_nums that have no more children (extinct species)
        for species_num in self.species:
            if not self.species[species_num]['children']:
                del self.species[species_num]
                del self.species_count[species_num]


    def simulate(self):
        self.sandboxes = [Sandbox(genome) for genome in self.genomes]
        for i, sandbox in enumerate(self.sandboxes):
            while True:
                try: # try to continue playing the game 
                    sandbox.set_input()
                    sandbox.make_next_move()
                    sandbox.reset_update()
                except:
                    print(f'Sandbox {i} finished with score of {sandbox.network.fitness}')
                    break # once a game has won, lost, or gotten stuck, break While loop, move onto new sandbox

    def adjust_fitness(self):
        '''
        Implements fitness sharing within species.
        Fitness sharing is important because it penalizes the fitness of individuals in large populations.
        This discourages one population to become dominant, which allows the algorithm to more effectively
        search the space of all topologies.
        '''
        for species_num in self.species:
            n = len(self.species[species_num]['children'])
            self.species_count[species_num] = n
            for children in self.species[species_num]['children']:
                children.fitness /= n

    
    def reproduce(self):
        '''
        The total adjusted fitness determines how many offsprings each species will get in the next generation
        '''
        next_gen: list [nn.NetworkGenome] = []

        species_fitness = {}
        for species_num in self.species:
            species_fitness[species_num] = sum([g.sum() for g in self.species[species_num]['children']])
            self.species[species_num]['children'].sort(reversed = True, key = lambda x: x.fitness)
            if self.species_count >= 5:
                next_gen.append(self.species[species_num]['children'][0])
        
        remaining = self.pop_size - len(next_gen)
        next_gen_pops = {}
        total_fitness = sum(species_fitness.values())

        for species_num in self.species:
            next_gen_pops[species_num] = int(species_fitness[species_num] / total_fitness * remaining)
        
        # ensure next gen has correct number of individuals
        while sum(next_gen_pops.values()) < remaining:
            next_gen_pops[random.choice(next_gen_pops.keys())] += 1

        for species in next_gen_pops:
            for _ in range(next_gen_pops[species]):
                # create an offspring by crossing over within this species
                # use roulette wheel selection



        # if a population has more than 5 individuals, the champion is included in the next generation unperturbed
        

    @staticmethod
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
