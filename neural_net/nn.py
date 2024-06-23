import random
import math


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
