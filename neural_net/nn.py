import random
import math
from .LSD import LSD
import pickle


from sim.constants import NEURON_ADD_CHANCE, SYNAPSE_WEIGHT_CHANGE_CHANCE, SYNAPSE_ADD_CHANCE

# Implements Neurons, Synapses, Layer, and Network

# the Neuron class
class Neuron:
    def __init__(self, id: int, bias: float = None, activation_f = None):
        self.id: int = id # innovation id of the neuron
        self.bias = bias # the bias of the neuron

        if activation_f is None:
            self.activation_f = 'sigmoid'
        else:
            self.activation_f = activation_f

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
        self.outof.connect(self)


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

    def list_activations(self): # utility function that prints out activation of layer in readable format
        return [f'neuron {neuron.id}: {neuron.value}' for neuron in self.neurons]
            
    def softmax(self): # perform a softmax operation of this layer
        bottom = sum(math.exp(neuron.get_activation()) for neuron in self.neurons)
        for neuron in self.neurons:
            neuron.value = math.exp(neuron.value) / bottom


class NeuronGene: # class for holding neuron information, but doesn't have neuron functionality
    def __init__(self, id: int, bias: float = None, activation_f = None):
        self.id = id
        self.bias = bias
        self.activation_f = activation_f
        self.out_synapses: list[SynapseGene] = []

        ################## not sure if this is the right approach
        self.expressed_neuron = None # <------------------------------------------------------------------------------------
    
    def express(self):
        self.expressed_neuron = Neuron(self.id, self.bias, self.activation_f)

    def connect(self, synapse):
        self.out_synapses.append(synapse)

    def __repr__(self):
        return f'[NeuronGene {self.id}] Bias: {self.bias}'
    
    # crossover only gets called when the id are the same
    @staticmethod
    def crossover(gene1: 'NeuronGene', gene2: 'NeuronGene'):
        if gene1.id != gene2.id:
            raise Exception(f'NeuronGene ids not matching: {gene1.id} != {gene2.id}')
        bias = random.choice([gene1.bias, gene2.bias])
        return NeuronGene(gene1.id, bias, gene1.activation_f)
    
    # used in LSD class to check equality between the original NeuronGene and a COPY of the same NeuronGene
    def id_eq(self, other):
        return self.id == other.id

class SynapseGene: # class for holding neuron information, but doesn't have synapse functionality
    def __init__(self, id: int, outof: NeuronGene, into: NeuronGene, weight: float, is_on: bool): ############## how to find neuron by IDs??
        self.id = id # innovation id of the synapse
        self.outof = outof # the origin neuron id of the synapse
        self.into = into # the destination neuron id of the synapse
        self.weight = weight
        self.is_on = is_on # if the synapse is disabled or not

        self.outof.connect(self)

    def express(self):
        # Synapse.__init__() automatically connects self to outof Neuron
        return Synapse(self.id, self.outof.expressed_neuron, self.into.expressed_neuron, self.weight, self.is_on)

    def __repr__(self):
        return f'[SynapseGene {self.id}] Neuron {self.outof.id} -> Neuron {self.into.id}, Weight: {self.weight}, {"Enabled" if self.is_on else "Disabled"}'
    
    @staticmethod
    def copy_update_synapse(old_synapse, new_outof, new_into):
        return SynapseGene(old_synapse.id, new_outof, new_into, old_synapse.weight, old_synapse.is_on)
    
    # crossover only gets called when the id are the same
    @staticmethod
    def crossover(gene1: 'SynapseGene', gene2: 'SynapseGene', child_neurons_ids: 'NetworkGenome.neuron_ids'):
        if gene1.id != gene2.id: # assert that ids are the same
            print(gene1)
            print(gene2)
            raise Exception(f'NeuronGene ids not matching: {gene1.id} != {gene2.id}')
        if gene1.outof.id != gene2.outof.id:
            print(gene1)
            print(gene2)
            raise Exception(f'NeuronGene OutOf ids not matching: {gene1.outof.id} != {gene2.outof.id}')
        if gene1.into.id != gene2.into.id:
            print(gene1)
            print(gene2)
            raise Exception(f'NeuronGene InTo ids not matching:  {gene1.into.id} != {gene2.into.id}')
        
        is_on = random.choice([gene1.is_on, gene2.is_on])
        weight = random.choice([gene1.weight, gene2.weight])

        into_id = gene1.into.id
        outof_id = gene1.outof.id
        child_into_neuron = child_neurons_ids[into_id]
        child_outof_neuron = child_neurons_ids[outof_id]

        return SynapseGene(gene1.id, child_outof_neuron, child_into_neuron, weight, is_on)

        
# This class holds all the genetic information of a network
# Such as information about deactivated synapses, all neurons, and all synapses
class NetworkGenome:
    # class attribute are accessible to read and update from all instances of NetworkGenome
    NIN = 0
    SIN = 0

    def __init__(self, input_neurons: list[NeuronGene], output_neurons: list[NeuronGene], neuron_gene: list[NeuronGene], synapse_gene: list[SynapseGene], dominant_parent: 'NetworkGenome' = None, recessive_parent: 'NetworkGenome' = None):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.neuron_gene = neuron_gene # excludes input and output layer genes
        self.synapse_gene = synapse_gene

        self.dominant_parent = None
        if dominant_parent is not None:
            self.dominant_parent = dominant_parent
            
        self.recessive_parent = None
        if recessive_parent is not None:
            self.recessive_parent = recessive_parent
        
        # these two are used in the synapse addition
        self.input_hidden_neurons = self.input_neurons + self.neuron_gene
        self.hidden_output_neurons = self.output_neurons + self.neuron_gene

        self.all_neuron_genes = self.input_neurons + self.output_neurons + self.neuron_gene

        self.neuron_ids = {n.id: n for n in self.all_neuron_genes}
        self.synapse_ids = {s.id: s for s in synapse_gene}

        self.fitness = 0

        # will be used to mark species #
        self.species = None

        # flag that gets used in simulation stage
        # if neural network is fittest individual, this flag will be set to False, and the network will be passed to the next generation without any mutations
        self.mutable = True

    def find_neuron(self, id):
        if id in self.neuron_ids:
            return self.neuron_ids[id]
        raise Exception(f'neuron id {id} not found')

    def find_synapse(self, id):
        if id in self.synapse_ids:
            return self.synapse_ids[id]
        raise Exception(f'synapse id {id} not found')
    
    def update_synapses(self, synapse_list: list[SynapseGene]): # ---------> not sure what this function was going to be for
        # self.synapse_gene
        return None
    
    # only does top sort on hidden + output neurons
    def top_sort(self, neuron_list) -> list[NeuronGene]:
        visited = set()
        sorted_neurons = []
        search_set = set(neuron_list)

        def dfs(visited_set: set, top_sort: list[NeuronGene], neuron: NeuronGene, search_set: list[NeuronGene]): # depth-first search method as helper
            for synapse in neuron.out_synapses:
                connected = synapse.into
                if connected not in visited_set and connected in search_set:
                    # need to check if connected in search_set, because we want to limit top sort of neurons contained in neuron_list
                    dfs(visited_set, top_sort, connected, search_set)
            visited_set.add(neuron)
            sorted_neurons.append(neuron)

        for neuron in neuron_list: # while all neurons haven't been discovered
            if neuron not in visited:
                dfs(visited, sorted_neurons, neuron, search_set)

        return sorted_neurons
     
    def mutate(self):
        # print('mutating')
        # synapse weight change
        for sg in self.synapse_gene:
            synapse_weight_change = random.random() <= SYNAPSE_WEIGHT_CHANGE_CHANCE
            # if synapse_switch:
            #     print(f'[synapse en/disable mutation] @ sg {sg.id}')
            #     sg.is_on = not sg.is_on
            if synapse_weight_change:
                new_weight = random.uniform(0, 1)
                # print(f'[synapse weight change mutation] @ sg {sg.id} ({sg.weight:.2f} -> {new_weight:.2f})')
                sg.weight = new_weight

        # neuron add
        # done by disabling an existing synapse, adding two new synapses, with a neuron in between
        neuron_add = random.random() <= NEURON_ADD_CHANCE

        if neuron_add:
            # pick a synapse at random
            to_disable = random.choice(self.synapse_gene)
            # disable it
            to_disable.is_on = False
            # make a new neuron
            # print('generating random bias')
            new_neuron = NeuronGene(NetworkGenome.NIN, random.uniform(-1, 1)) # <----- need mechanism to keep track of IDs
            # print(f'[neuron addition mutation] [Neuron {NetworkGenome.NIN}] @ sg {to_disable.id}, bias = {new_neuron.bias}')
            NetworkGenome.NIN += 1
            self.update_neuron_lists(new_neuron)

            # create two new synapses, and connect the new neuron with it
            # print(f'to_disable.outof: {to_disable.outof}')
            # print(f'to_disable.into: {to_disable.into}')
            new_synapse_into = SynapseGene(NetworkGenome.SIN, to_disable.outof, new_neuron, 1, True)
            NetworkGenome.SIN += 1
            new_synapse_outof = SynapseGene(NetworkGenome.SIN, new_neuron, to_disable.into, to_disable.weight, True)
            NetworkGenome.SIN += 1

            self.update_synapse_lists(new_synapse_into)
            self.update_synapse_lists(new_synapse_outof)

        # synapse add
        # must ensure new synapse does not make a loop
        
        synapse_add = random.random() <= SYNAPSE_ADD_CHANCE

        # what operation do i need?
            # need efficient set DIFFERENCE operation to ensure there are available nodes for being into_neuron (allowed neurons)
            # need efficient SAMPLING (random.choice) to randomly pick into_neuron (allowed neurons)
            # need efficient set REMOVAL operation in case this outof_neuron is invalid (candidates list, has to be ordered) 

        if synapse_add:
            attempts = 3
            sorted_neurons = list(reversed(self.top_sort(self.hidden_output_neurons))) # top sort of hidden
            sorted_neurons_d = {n: i for (n, i) in zip(sorted_neurons, range(len(sorted_neurons)))}
            sorted_neurons_set = set(sorted_neurons)
            # outof_candidates = list(range(len(self.input_hidden_neurons) - 1))
            hidden_outof_candidates = LSD(self.neuron_gene)
            input_outof_candidates = LSD(self.input_neurons)
            # print(outof_candidates)
            
            allowed_neurons = set()
            outof_neuron = None
            while True: # until a valid new connection is found
                if len(hidden_outof_candidates) + len(input_outof_candidates) == 0: # if there are no more source neuron candidates, give up adding a synapse
                    break

                # determine if outof neuron is going to be hidden or input neurons
                from_input_chance = len(self.input_neurons) / len(self.input_hidden_neurons)
                from_input = random.random() <= from_input_chance

                if from_input: # from input neuron -> hidden neuron or output neuron
                    # outof_neuron_id = random.choice(input_outof_candidates)
                    # outof_neuron = self.input_neurons[outof_neuron_id] # pick a random input neuron
                    outof_neuron = input_outof_candidates.sample()
                    allowed_neurons = sorted_neurons_set - set([s.into for s in outof_neuron.out_synapses])

                    if not allowed_neurons:
                        input_outof_candidates.remove(outof_neuron)

                else: # from hidden neuron -> later hidden neuron or output neuron
                    outof_neuron = hidden_outof_candidates.sample()
                    # print(hidden_outof_candidates)
                    # print(sorted_neurons_d)
                    outof_neuron_idx = sorted_neurons_d[outof_neuron]
                    later_neurons = sorted_neurons[outof_neuron_idx + 1:] # neurons that topologically come after outof_neuron
                    allowed_neurons = set(later_neurons) - set([s.into for s in outof_neuron.out_synapses])

                    if not allowed_neurons:
                        hidden_outof_candidates.remove(outof_neuron)
                
                if allowed_neurons:
                    into_neuron = random.choice(list(allowed_neurons))
                    new_synapse = SynapseGene(NetworkGenome.SIN, outof_neuron, into_neuron, random.uniform(-1, 1), True)
                    NetworkGenome.SIN += 1
                    self.update_synapse_lists(new_synapse)
                    # print(f'[synapse addition mutation] {new_synapse}')
                    break
                
                attempts -= 1
                if attempts == 0:
                    # print('3 attempts were made at creating a new synapse, but failed')
                    break

    # utility function to update all the lists of NeuronGene
    def update_neuron_lists(self, new_neuron: NeuronGene):
        self.neuron_gene.append(new_neuron)
        self.input_hidden_neurons.append(new_neuron)
        self.hidden_output_neurons.append(new_neuron)
        self.all_neuron_genes.append(new_neuron)

        self.neuron_ids[new_neuron.id] = new_neuron

    # same utility function
    def update_synapse_lists(self, new_synapse: SynapseGene):
        self.synapse_gene.append(new_synapse)

        self.synapse_ids[new_synapse.id] = new_synapse

    @classmethod
    def from_crossover(cls, parent1, parent2):
        # assigns dominant to fitter Network
        dominant, recessive = None, None
        dominant, recessive = (parent1, parent2) if parent1.fitness > parent2.fitness else (parent2, parent1)

        # print(f'{parent1, parent1.fitness, parent2, parent2.fitness}')

        neuron_gene = []
        input_l_genes = []
        output_l_genes = []
        synapse_gene = []

        # crossover neuron_genes
        for ng in dominant.neuron_gene:
            if ng.id in recessive.neuron_ids: 
                # if this NeuronGene is in the recessive NetworkGenome, do crossover
                hybrid_neuron = NeuronGene.crossover(ng, recessive.find_neuron(ng.id))
                neuron_gene.append(hybrid_neuron)
            else:
                # if this neuron is only present in the dominant, add it to the genome
                # new_neuron = copy.deepcopy(ng)
                new_neuron = NeuronGene(ng.id, ng.bias, ng.activation_f)
                neuron_gene.append(new_neuron) 
        # print(neuron_gene)

        # crossover input layer neurons
        for ng in dominant.input_neurons:
            hybrid_neuron = NeuronGene.crossover(ng, recessive.find_neuron(ng.id))
            input_l_genes.append(hybrid_neuron)

        # crossover output layer neurons
        for ng in dominant.output_neurons:
            hybrid_neuron = NeuronGene.crossover(ng, recessive.find_neuron(ng.id))
            output_l_genes.append(hybrid_neuron)

        # initialize child NetworkGenome so we can access various NetworkGenome attributes
        # need child.neuron_ids attribute for the synapse crossover, because synapses must be connected
        # for example, having access to child.neuron_ids makes crossing over synapses easier
        child = cls(input_l_genes, output_l_genes, neuron_gene, synapse_gene, dominant, recessive)

        for sg in dominant.synapse_gene:
            if sg.id in recessive.synapse_ids:
                # if this SynapseGene is in the recessive NetworkGenome, do crossover
                hybrid_synapse = SynapseGene.crossover(sg, recessive.find_synapse(sg.id), child.neuron_ids)
                child.synapse_gene.append(hybrid_synapse)
            else:
                # if this synapse is only present in the dominant, create new genome with same parameters, add it to the genome
                into_id = sg.into.id
                outof_id = sg.outof.id
                child_into_neuron = child.find_neuron(into_id)
                child_outof_neuron = child.find_neuron(outof_id)
                new_synapse = SynapseGene.copy_update_synapse(sg, child_outof_neuron, child_into_neuron)
                child.synapse_gene.append(new_synapse)

        # need to manually run this to ensure child genome has synapse_gene dict
        child.synapse_ids = {s.id: s for s in child.synapse_gene}

        return child

    @staticmethod
    # @profile
    def distance(net1: 'NetworkGenome', net2: 'NetworkGenome', w_disjoint :float, w_excess :float, w_weight :float, threshold :float):
        N = max(len(net1.synapse_gene), len(net2.synapse_gene))

        # set1 = set([sg.id for sg in net1.synapse_gene])
        # set2 = set([sg.id for sg in net2.synapse_gene])

        max_net1 = net1.synapse_gene[-1].id
        max_net2 = net2.synapse_gene[-1].id

        disjoint, excess, common, weight_diff = 0, 0, 0, 0

        for sg in net1.synapse_gene:
            # if sg.id not in set2:
            if sg.id not in net2.synapse_ids:
                if sg.id < max_net2:
                    disjoint += 1
                else:
                    excess += 1
            else:
                weight_diff += abs(sg.weight - net2.find_synapse(sg.id).weight)
                common += 1

        w_bar = weight_diff / common
        # implementing early stopping
        delta = disjoint * w_disjoint + excess * w_excess + w_bar * w_weight

        if delta >= threshold:
            return delta

        for sg in net2.synapse_gene:
            # if sg.id not in set1:
            if sg.id not in net1.synapse_ids:
                if sg.id < max_net1:
                    disjoint += 1
                else:
                    excess += 1
        

        delta = disjoint * w_disjoint + excess * w_excess + w_bar * w_weight
        
        return delta
    
    @staticmethod
    def save_genome(genome: 'NetworkGenome', fp: str):
        with open(fp, 'wb') as file:
            pickle.dump(genome, file)
    
    @staticmethod
    def load_genome(fp: str):
        with open(fp, 'rb') as file:
            genome = pickle.load(file)
        return genome



    def __repr__(self):
        return f'Neural Network Genome with {len(self.neuron_ids)} neurons, {len(self.synapse_ids)} synapses'

    

class Network:
    #################### change so that initializes from NetworkGenome instance
    def __init__(self, genome: NetworkGenome):
        # first, express all neurons in genome
        for neuron_gene in genome.all_neuron_genes:
            neuron_gene.express() # -----> this sets neuron_gene.expressed_neuron to an actual Neuron
        
        self.input_l = Layer([n.expressed_neuron for n in genome.input_neurons])
        self.output_l = Layer([n.expressed_neuron for n in genome.output_neurons])
        self.genome = genome

        self.fitness = 0 # used when there's conflicting genes during crossover
        self.neurons = self.input_l.neurons + self.output_l.neurons + [n.expressed_neuron for n in genome.neuron_gene]

        self.sorted_neurons = list[Neuron]

        self.synapses = [s.express() for s in genome.synapse_gene]

    def set_input(self, inputs: list[float]): # sets the input neurons' values
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

    def flush_values(self):
        for neuron in self.sorted_neurons:
            neuron.flush_value()

    # utility functions
    def add_hidden_neuron(self, neuron: Neuron): # unused function, because neurons will not be added to networks, but genome instead
        self.neurons.append(neuron)

    def add_hidden_synapse(self, synapse: Synapse): # likewise to add_hidden_neuron
        self.synapses.append(synapse)

    def __repr__(self):
        return f'Neural Network with {len(self.neurons)} neurons, {len(self.synapses)} synapses'
    
    def set_fitness(self, fitness):
        self.fitness = fitness
        self.genome.fitness = fitness

    def play_game(self):
        pass