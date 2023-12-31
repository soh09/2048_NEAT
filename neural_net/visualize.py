from graphviz import Digraph
import nn

# function to visualize neural network
def visualize_neural_network(network: nn.Network, orientation = 'LR', plot_genome = False):
    dot = Digraph(comment='Neural Network', format='png')
    dot.attr(rankdir=orientation)  # Set the rank direction to left to right

    for neuron in network.neurons:
        if isinstance(neuron, nn.NeuronGene): # if this is a NetworkGenome visualization, NeuronGene have no value
            label = f'#{neuron.id} | Bias={neuron.bias:.2f}'
        else:
            label = f'#{neuron.id} | Bias={neuron.bias:.2f}\nval={neuron.value:.4f}'
        dot.node(f'Neuron_{neuron.id}', label=label, shape='ellipse', width='0.1', height='0.5', fontsize='10')

        for synapse in neuron.out_synapses:
            if synapse.is_on:
                # label = f'w={synapse.weight:.2f}'
                label = f'{synapse.id}'
                dot.edge(f'Neuron_{neuron.id}', f'Neuron_{synapse.into.id}', label=label, fontsize = '10')
            elif plot_genome:
                label = f'w={synapse.weight:.2f}'
                dot.edge(f'Neuron_{neuron.id}', f'Neuron_{synapse.into.id}', label=label, fontsize = '10', color = '#ff000040', fontcolor = '#ff000040')

    with dot.subgraph() as s:
        s.attr(rank='same')
        for input_n in network.input_l.neurons:
            s.node(f'Neuron_{input_n.id}')

    with dot.subgraph() as s:
        s.attr(rank='same')
        for output_n in network.output_l.neurons:
            s.node(f'Neuron_{output_n.id}')

    return dot