from Sandbox import Sandbox as Sandbox
import neural_net.nn as nn
import random
from math import e
from copy import deepcopy
from constants import POP_SIZE, SPECIATION_THRESHOLD, W_DISJOINT, W_EXCESS, W_WEIGHT, KILL_SPECIES_AFTER_NO_IMPROVEMENTS, REWARD_TYPE
import time
from datetime import datetime
import os
import json
import psutil
import gc

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
            w_weight: float = W_WEIGHT,
            reward_type: str = REWARD_TYPE,
            log_folder: str = None,
            debug = False):
        
        self.pop_size = population
        self.speciation_threshold = spec_thres
        self.w_disjoint = w_disjoint
        self.w_excess = w_excess
        self.w_weight = w_weight
        self.reward_type = reward_type

        self.species_counter = 0
        
        self.current_gen = 0

        self.log = {
            'current_gen': 0,
            'avg_fitness': 0,
            'max_fitness': 0,
            'sim_time': 0,
            'mutate_time': 0,
            'alive_species': 0,
            'stagnant_species': 0,
            'species_list': [],
            'RSS': 0,
            'VMS': 0,
            'gc_gen0': 0,
            'gc_gen1': 0,
            'gc_gen2': 0
        }
        
        self.debug = debug
        self.log_folder = log_folder
        self.log_file = None
        if self.log_folder:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.v2.txt'
            self.log_file = os.path.join(self.log_folder, timestamp)

        
        # self.species structure
        # self.species = {
        #     species_num: {
        #         'progenitor': nn.NetworkGenome,
        #         'children': list [nn.NetworkGenome]
        #         'stats': [generation where max fitness was seen, max fitness]
        #     }
        # }
        self.species = {}

        self.species_size: dict [int, int] = {}
        self.genomes: list [nn.NetworkGenome] = []
        self.sandboxes: list [Sandbox] = []

        # initialize population
        for i in range(population):
            # print('~~checking new genome~~')
            genome = Simulation.create_dense_network()
            # net = nn.Network(genome)
            self.genomes.append(genome)

            # if first genome, that will automatically be the progenitor 
            if i == 0:
                # progenitor has to be deepcopy, because this genome will be modified in-place later when mutated
                self.species[self.species_counter] = {'progenitor': deepcopy(genome.synapse_gene), 'children': [genome], 'stats': [self.current_gen, 0]}
                genome.species = 0
                self.species_counter += 1
            else:
                # print(self.species_counter)
                new_species = True
                # even though the rest should be classified as the same species, well check just in case
                for species_num in self.species:
                    # print(f'checking {species_num}')
                    dist = nn.NetworkGenome.distance(self.species[species_num]['progenitor'], genome, self.w_disjoint, self.w_excess, self.w_weight, self.speciation_threshold)
                    # print(dist)
                    if dist < self.speciation_threshold: # if genome is the same species as species_num
                        new_species = False
                        self.species[species_num]['children'].append(genome)
                        genome.species = species_num
                        break
                if new_species:
                    # print('new species')
                    self.species[self.species_counter] = {'progenitor': deepcopy(genome.synapse_gene), 'children': [genome], 'stats': [self.current_gen, 0]}
                    genome.species = self.species_counter
                    self.species_counter += 1
        for species_num in self.species:
            n = len(self.species[species_num]['children'])
            self.species_size[species_num] = n


###############################################
        # Enable debugging with verbosity
        gc.set_debug(gc.DEBUG_UNCOLLECTABLE)

        # Initialize counters for each generation
        self.deallocated_objects = {0: 0, 1: 0, 2: 0}

        # Callback function to tally deallocated objects
        def gc_callback(phase, info):
            if phase == "stop":  # Only tally after GC has finished
                gen = info['generation']
                self.deallocated_objects[gen] += info['collected']

        # Attach the callback to GC events
        gc.callbacks.append(gc_callback)

################################################

        print(f'{POP_SIZE} NetworkGenomes created, ready for simulation')

    # @profile
    def mutate_and_speciate(self):
        now = time.time()

        # TODO
        # might be worth investigating if = [] is enough to free up memory
        # reset self.species
        for species_num in self.species:
            self.species[species_num]['children'] = []
        
        # mutate genomes
        for genome in self.genomes:
            if genome.mutable:
                genome.mutate()
            else:
                # set mutable to True for this generations, so it can be mutated in future generations
                genome.mutable = True

            # then put in appropriate species
            new_species = True
            for species_num in self.species:
                # print(f'checking {species_num}')
                dist = nn.NetworkGenome.distance(self.species[species_num]['progenitor'], genome, self.w_disjoint, self.w_excess, self.w_weight, self.speciation_threshold)
                # print(dist)
                if dist < self.speciation_threshold: # if genome is the same species as species_num
                    new_species = False
                    self.species[species_num]['children'].append(genome)
                    genome.species = species_num
                    break
            if new_species:
                # print('new species')
                self.species[self.species_counter] = {'progenitor': deepcopy(genome.synapse_gene), 'children': [genome], 'stats': [self.current_gen, 0]}
                genome.species = self.species_counter
                self.species_counter += 1

        # remove species_nums that have no more children (extinct species)
        current_species = list(self.species.keys())
        for species_num in current_species:
            if not self.species[species_num]['children']:
                del self.species[species_num]
                del self.species_size[species_num]

        self.log['mutate_time'] = time.time() - now

        self.create_log()


    # @profile
    def simulate(self):
        now = time.time()

        self.deallocated_objects = {0: 0, 1: 0, 2: 0}

        # path = '/Users/so/Documents/projects/personal/2048_AI/logs/nn_init.txt'
        # with open(path, 'a') as log:
        #     log.write(f'{self.current_gen}, ')

        counts = f'{self.current_gen}, '
        self.sandboxes = []
        for genome in self.genomes:
            # nc, sc = len(genome.neuron_ids), len(genome.synapse_ids)
            # counts += f'{nc}, {sc}, '
            net = nn.Network(genome)
            sb = Sandbox(net, self.debug)
            self.sandboxes.append(sb)
        # self.sandboxes = [Sandbox(nn.Network(genome), self.debug) for genome in self.genomes]

        # path = '/Users/so/Documents/projects/personal/2048_AI/logs/nn_init.txt'
        # with open(path, 'a') as log:
        #     log.write(f'\n')

        # print(f'sandbox creation: {(time.time() - now):.3f}')
        max_fitness = 0
        total_fitness = 0
        while_start = time.time()
        for i, sandbox in enumerate(self.sandboxes):
            # print(f'individual {i}')
            while True:
                try: # try to continue playing the game 
                    sandbox.set_input()
                    sandbox.make_next_move(self.reward_type)
                    sandbox.reset_update()
                except Exception as e:
                    total_fitness += sandbox.network.fitness
                    if max_fitness < sandbox.network.fitness:
                        max_fitness = sandbox.network.fitness

                    # potentially update the max fitness for that species
                    if sandbox.network.fitness > self.species[sandbox.network.genome.species]['stats'][1]:
                        self.species[sandbox.network.genome.species]['stats'][0] = self.current_gen
                        self.species[sandbox.network.genome.species]['stats'][1] = sandbox.network.fitness
                    # print(f'Sandbox {i} finished with score of {sandbox.network.fitness}')
                    break # once a game has won, lost, or gotten stuck, break While loop, move onto new sandbox
        
        # print(f'make next move: {(time.time() - while_start):.3f}')

        total = time.time() - now

        # get memory info
        process = psutil.Process()
        mem_info = process.memory_info()

        print(f'(max, avg) unadjusted fitness of generation {self.current_gen} = {(max_fitness, total_fitness / POP_SIZE)}')
        self.log['current_gen'] = self.current_gen
        self.log['sim_time'] = total
        self.log['max_fitness'] = max_fitness
        self.log['avg_fitness'] = total_fitness / POP_SIZE
        self.log['RSS'] = mem_info.rss
        self.log['VMS'] = mem_info.vms
        self.log['gc_gen0'] = self.deallocated_objects[0]
        self.log['gc_gen1'] = self.deallocated_objects[1]
        self.log['gc_gen2'] = self.deallocated_objects[2]
        self.current_gen += 1

    def adjust_fitness(self):
        '''
        Implements fitness sharing within species.
        Fitness sharing is important because it penalizes the fitness of individuals in large populations.
        This discourages one population to become dominant, which allows the algorithm to more effectively
        search the space of all topologies.
        '''
        for species_num in self.species:
            n = len(self.species[species_num]['children'])
            self.species_size[species_num] = n
            # print(f'species #{species_num} has {n} individuals')
            for children in self.species[species_num]['children']:
                # print(f'og fitness: {children.fitness}')
                children.fitness /= n
                # print(f'new fitness: {children.fitness}')
    def reproduce(self):
        '''
        The total adjusted fitness determines how many offsprings each species will get in the next generation
        '''
        next_gen: list [nn.NetworkGenome] = []

        # check for stagnant species, and kill that species off
        current_species = list(self.species.keys())
        killed = 0
        for species_num in current_species:
            if self.species[species_num]['stats'][0] < self.current_gen - KILL_SPECIES_AFTER_NO_IMPROVEMENTS:
                if len(self.species) >= 5: # ensure that there are at least 5 species in the population
                    killed += 1
                    del self.species[species_num]
                    del self.species_size[species_num]

        print(f'total # of species: {len(current_species)}, # of stagnant species: {killed}')



        species_fitness = {}
        for species_num in self.species:
            species_fitness[species_num] = sum([g.fitness for g in self.species[species_num]['children']])
            self.species[species_num]['children'].sort(reverse = True, key = lambda x: x.fitness)
            # fittest individual gets added to new generation without any modifications
            if self.species_size[species_num] >= 5:
                # set mutable to False to prevent mutation in next generation
                self.species[species_num]['children'][0].mutable = False
                next_gen.append(self.species[species_num]['children'][0])
        
        # do inter-species mating for 3% of the population
        species_list = list(self.species.keys())
        print(species_list)
        for _ in range(int(POP_SIZE * 0.03)):
            parent1_species, parent2_species = random.choices(species_list, k = 2)
            parent1 = random.choice(self.species[parent1_species]['children'])
            parent2 = random.choice(self.species[parent2_species]['children'])
            offspring = nn.NetworkGenome.from_crossover(parent1, parent2)
            next_gen.append(offspring)

        
        remaining = self.pop_size - len(next_gen)
        species_allocation = {}
        total_fitness = sum(species_fitness.values())

        for species_num in self.species:
            species_allocation[species_num] = int(species_fitness[species_num] / total_fitness * remaining)

        # ensure next gen has correct number of individuals
        while sum(species_allocation.values()) < remaining:
            species_allocation[random.choice(list(species_allocation.keys()))] += 1

        for species_num in species_allocation:
            # create an offspring by crossing over within this species
            # use ~~roulette wheel selection~~ rank based
            species_n = self.species_size[species_num]
            # print(f'species #{species_num}, popsize = {species_n}, allocation = {species_allocation[species_num]}')
            proportional_prob = [e**(rank / species_n) for rank in range(species_n, 0, -1)]
            # print(f'probabilities: {proportional_prob}')
            # print(f'fitnesses: {[n.fitness for n in self.species[species_num]['children']]}')
            for _ in range(species_allocation[species_num]):
                parent1, parent2 = random.choices(self.species[species_num]['children'], weights = proportional_prob, k = 2)
                offspring = nn.NetworkGenome.from_crossover(parent1, parent2)
                next_gen.append(offspring)
        # update self.genomes + clear self.sandboxes
        self.genomes = next_gen
        self.sandboxes = []

        self.log['alive_species'] = len(current_species)
        self.log['stagnant_species'] = killed
        self.log['species_list'] = ', '.join([str(s) for s in species_list])



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

    def create_log(self):
        '''
        information to log:
        current_gen, avg_fitness, max_fitness, sim_time, mutate_time, 
        '''
        if self.log_folder:
            with open(self.log_file, 'a') as log:
                log.write(json.dumps(self.log) + '\n')
