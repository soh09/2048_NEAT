from graphviz import Digraph
import neural_net.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# function to visualize neural network
def visualize_net(network: nn.Network, orientation = 'LR', plot_genome = False):
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
def visualize_genome(genome, orientation='BT', show_label = True, ranksep = '1'):
    dot = Digraph(comment='Neural Network Genome', format='png')
    dot.attr(rankdir=orientation, ranksep = ranksep)

    for neuron in genome.all_neuron_genes:
        label = f'#{neuron.id} | Bias={neuron.bias:.2f}'
        if show_label:
            dot.node(f'Neuron_{neuron.id}', label=label, shape='ellipse', width='0.1', height='0.5', fontsize='10')
        else:
            dot.node(f'Neuron_{neuron.id}', shape='ellipse', width='0.1', height='0.5', fontsize='10')

    for synapse in genome.synapse_gene:
        label = f'id={synapse.id}, w={synapse.weight:.2f}'
        color = 'red' if not synapse.is_on else 'green'
        if show_label:
            dot.edge(f'Neuron_{synapse.outof.id}', f'Neuron_{synapse.into.id}', label=label, fontsize='10', color=color)
        else:
            dot.edge(f'Neuron_{synapse.outof.id}', f'Neuron_{synapse.into.id}', color=color)

    with dot.subgraph() as s:
        s.attr(rank='same')
        sorted_input_neurons = sorted(genome.input_neurons, key=lambda n: n.id)
        for input_n in sorted_input_neurons:
            s.node(f'Neuron_{input_n.id}', fontsize='10')

    with dot.subgraph() as s:
        s.attr(rank='same')
        sorted_output_neurons = sorted(genome.output_neurons, key=lambda n: n.id)
        for output_n in sorted_output_neurons:
            s.node(f'Neuron_{output_n.id}', fontsize='10')

    return dot


# this code is from GPT-4o, you are amazing.
def display_gene(synapse_genes):

    # Extracting all innovation IDs and determining the range
    ids = [sg.id for sg in synapse_genes]
    min_id = min(ids)
    max_id = max(ids)
    print(max_id)

    fig, ax = plt.subplots(figsize=(max_id * 0.8, 0.8))

    # Mapping each id to its corresponding synapse gene
    id_to_synapse = {sg.id: sg for sg in synapse_genes}

    # Setting the axis limits
    ax.set_xlim(0, max_id - min_id + 1)
    ax.set_ylim(0, 1)

    # Hiding the axes
    ax.axis('off')

    for i in range(min_id, max_id + 1):
        # Create a rectangle
        rect = patches.Rectangle((i - min_id, 0), 1, 1, edgecolor='black', facecolor='white')
        ax.add_patch(rect)

        if i in id_to_synapse:
            sg = id_to_synapse[i]
            # Add text for the innovation ID at the top
            ax.text(i - min_id + 0.5, 0.75, str(sg.id), ha='center', va='center', fontsize=10)

            # Add text for the neurons it connects
            ax.text(i - min_id + 0.5, 0.5, f"{sg.outof.id} -> {sg.into.id}", ha='center', va='center', fontsize=10)

            # Indicate if the synapse is disabled
            if not sg.is_on:
                ax.text(i - min_id + 0.5, 0.25, "DIS", ha='center', va='center', fontsize=10, color='gray')
        else:
            # Empty box for skipped innovation numbers
            ax.text(i - min_id + 0.5, 0.5, "", ha='center', va='center', fontsize=10)

    plt.show()