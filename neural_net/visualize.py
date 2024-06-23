from graphviz import Digraph
import nn_old as nn_old

# function to visualize neural network
def visualize_net(network: nn_old.Network, orientation = 'LR', plot_genome = False):
    dot = Digraph(comment='Neural Network', format='png')
    dot.attr(rankdir=orientation)  # Set the rank direction to left to right

    for neuron in network.neurons:
        if isinstance(neuron, nn_old.NeuronGene): # if this is a NetworkGenome visualization, NeuronGene have no value
            label = f'#{neuron.id} | Bias={neuron.bias:.2f}'
        else:
            label = f'#{neuron.id} | Bias={neuron.bias:.2f}\nval={neuron.value:.4f}'
        dot.node(f'Neuron_{neuron.id}', label=label, shape='ellipse', width='0.1', height='0.5', fontsize='10')

        for synapse in neuron.out_synapses:
            if synapse.is_on:
                label = f'id={synapse.id}, w={synapse.weight:.2f}'
                # label = f'{synapse.id}'
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

# function to visualize neural network genome
def visualize_genome(genome, orientation='BT'):
    dot = Digraph(comment='Neural Network Genome', format='png')
    dot.attr(rankdir=orientation)

    for neuron in genome.all_neuron_genes:
        label = f'#{neuron.id} | Bias={neuron.bias:.2f}'
        dot.node(f'Neuron_{neuron.id}', label=label, shape='ellipse', width='0.1', height='0.5', fontsize='10')

    for synapse in genome.synapse_gene:
        label = f'id={synapse.id}, w={synapse.weight:.2f}'
        color = 'red' if not synapse.is_on else 'green'
        dot.edge(f'Neuron_{synapse.outof.id}', f'Neuron_{synapse.into.id}', label=label, fontsize='10', color=color)

    with dot.subgraph() as s:
        s.attr(rank='same')
        for input_n in genome.input_neurons:
            s.node(f'Neuron_{input_n.id}')

    with dot.subgraph() as s:
        s.attr(rank='same')
        for output_n in genome.output_neurons:
            s.node(f'Neuron_{output_n.id}')

    return dot
