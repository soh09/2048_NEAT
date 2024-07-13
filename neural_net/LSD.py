# what operation do i need?
    # need efficient set DIFFERENCE operation to ensure there are available nodes for being into_neuron (allowed neurons)
    # need efficient SAMPLING (random.choice) to randomly pick into_neuron (allowed neurons)
    # need efficient set REMOVAL operation in case this outof_neuron is invalid (candidates list, has to be ordered)

# This class was created as a way to meet my needs in NetworkGenome.mutate() method

import copy
import random

class LSD: # LSD = List Set Dictionary (because it uses all these data structures)
    def __init__(self, lst):
        self.node_set = set(lst)
        self.node_list = copy.copy(lst)
        self.node_to_index = {node: i for (node, i) in zip(self.node_list, list(range(len(lst))))}

    def remove(self, node):
        if node not in self.node_set:
            raise Exception('node not in set')
        
        # removing from node_set
        self.node_set.discard(node)
        last_node = self.node_list.pop()

        if not node.id_eq(last_node):
            index = self.node_to_index[node]
            self.node_list[index] = last_node # move node to position index in node_list
            self.node_to_index[last_node] = index # update node_to_index[last_node] = index
        
        # deleting entry for node in node_to_index
        del self.node_to_index[node]

    def diff(self, other: set):
        return self.node_set - other
    
    def sample(self):
        if not self.node_list:
            raise Exception('node_list is empty')
        return random.choice(self.node_list)
    
    def __len__(self):
        return len(self.node_list)
    
    def __str__(self):
        return '[' + ', '.join(str(node) for node in self.node_list) + ']'

    def __repr__(self):
        return '[' + ', '.join(str(node) for node in self.node_list) + ']'