# So-NEAT 
Excuse the pun. This is a work in progress.

# Goal of Project
I want to create an neural network that can play the classic game 2048. I saw a
person playing this game in front of me in class today, and they were really
bad at it. They didn't know the strategy where you never press one of the directions.
This got me curious how AI could teach itself to play 2048. I've seen Youtube
videos where people make AI walk or play other games like pacman or snake. I wanted
to learn how genetic algorithms work as well, so this was a perfect small little
project for me to work on. Implementing a neural network from scratch should also
be insightful for me.

# Roadmap

## The game component
1. Borrow code from other 2048 recreation repos: DONE
    - [x] courtesy of github repo: https://github.com/yangshun/2048-python
    - thank you very much for your code. I will take good care of it.
2. Integrate the 2048 game with the model
    - [x] enable ai model to understand the current game state
        - what numbers there are, how many spaces are left open, etc
        - or just straight up the numbers of each square, since there aren't that many 
    - [x] enable model to control the game
        - simple key inputs (arrow keys most likely)

## The neural network + NEAT component
1. Implement a simple neural network
    - [x] create a neuron class
    - [x] create a synapse class
    - [x] implement genotype functionality
    - [x] implement forward pass capability
        - "forward pass" should forward the activation of a neuron to connected neuron
    - [x] create a layer class
        - Layer class will be useful for the input and output layers of the network, since the 
        number of neurons are fixed for these two cases
        - will have softmax capability to generate probabilities of each move
    - ~~create a multi-layer perceptron class~~ not necessary, since the point of NEAT is to create flexible topologies
    - [x] implement network class
        - similar to a MLP, but because of NEAT, doesn't have hidden layers, but
        has hidden neurons instead
        - [x] implement ability to forward pass on hidden neurons
            - use graph data structure for hidden neurons
            - [x] implement topological search for forward pass capability

2. Add NEAT functionality to step 1
    - [x] implement crossover (mating) functionality: IMPLEMENTING
        - allow two ways of instantiating a NetworkGenome class: from manual inputs, and from two NetworkGenomes crossover
        - probably want to weigh the dominant gene more than the recessive gene when doing a hybrid crossover
    - [x] Add NeuronGene, SynapseGene, NetworkGenome classes for holding genetic information
    - [x] implement building a model (phenotype) from the genotype
        - A Network instance will be the phenotypic expression of a NetworkGene instance
        - similar to an inheritance structure
    - [x] implement mutation
        - make sure no cycles occur as a result
            - perform top sort, choose source neuron, then choose destination neuron in the later order
        - [x] implement data structure that facilitates quick validation of potential mutation (LSD class)
            
        - types of mutations
            - neuron addition
                - neuron addition when there is an new id?
            - [x] synapse addition
                - for the source neuron, we want to choose from input neurons and hidden neurons
                - for the destination neuron, we want to choose from hidden or output neurons
                - chosen synapse must not create a loop, and ideally source neuron should be lower order in a topological sort
            - neuron bias change (low priority)
            - synapse weight change 
    - [ ] implement some sort of species differentiating algorithm
    - [ ] implement logic for evolving networks
        - when to "kill" certain underperforming species
        - probabilities for sexual reproduction, asexual reproduction, etc

3. Simulation of Generations and Populations
    - [x] create Sandbox class, which takes in a Network class, instantiates a Game object, and makes it play till game over/win condition is met

### To Do
- [ ] think about neuron and synapse cross over probabilities (dominant vs recessive)
- [ ] add a nice visual for the structure of the classes, methods, and attributes
- [ ] add nicer comments
- [x] fix bug where there is duplicate synapses sometimes. figure out where it comes from
- [x] think about how to have a global counter variable for innovation number

## Some Technical Considerations
### Neural Network
- Layer class .forward() function will be used for the input layer
- Layer class .softmax() function will be used for the output layer
- For the forward pass for hidden neurons, the topological sort will be
performed on the Synapses (the links), not the neurons. 
- Used depth-first search to perform a topological sort on the neurons to ensure
the forward pass is done properly
    - ie, all input neurons must be forward-passed for the output neuron to forward
- depth-first search is used for creating new connections as a mutation, to ensure no cycles emerge
    - with this approach, a hidden neuron can feed back into a input neuron, should I allow this?

## Progression
| date  | details |
| ----- | ------- |
| 11/23 | Project started |
| 11/24 ~ 11/26 | Implemented Neuron, Synapse, Network classes. Implemented forward pass capability. |
| 11/27 | Started writing Genome Classes (NeuronGene, SynapseGene, NetworkGenome). Started implemented crossover logic. | 
| 11/30 | Started implementing crossover logic. Started logic for NetworkGenome expression into concrete Network class. |
| 12/7  | Working on implementing crossover logic. Finished logic forNetworkGenome expression into concrete Network class. Updated layout of README. |
| 12/31 | Implementing and testing crossover logic. Improved visualize_neural_network function. |
|  1/1  | Implementing mutation logic. |
|  6/19 | REBOOTING PROJECT. Reviewing Neural Net code.| 
|  6/23 | Implemented LSD class, which implements quick removal, set difference, and sampling. |
|  7/1  | I think I fixed the synapse addition logic FINALLY!!! |
|  7/7  | Added innovation ID logic. Added Gene visualization like in the paper.|
|  7/13 | Implemented distance function. Started work on game code to make neural network be able to play it, called Sandbox. |
|  7/20 | Started development of simulation. Testing with no selection logic yet, population = 1000. Random mutations every iterations. |
|  7/21 | Updated neuron bias logic. Added pickling support to NetworkGenome. Started working on Simulation class. Added constants.py. |

# Attribution

## 2048-python
Github repo: https://github.com/yangshun/2048-python<br>
Contributors: [Yanghun Tay](http://github.com/yangshun), [Emmanuel Goh](http://github.com/emman27)

## General Resources
- <a href = 'https://www.youtube.com/watch?v=lAjcH-hCusg'>NEAT algorithm from scratch (it was hard)</a> by Tech with Nikola
    - For inspiration on how to code NEAT
- <a href = 'https://www.youtube.com/watch?v=b3D8jPmcw-g'>Neuroevolution of Augmenting Topologies (NEAT)</a> by Connor Shorten
    - For understanding how NEAT works
- <a href = 'https://github.com/karpathy/micrograd/tree/master'>micrograd Neural Network Library</a> by Andrej Karpathy
    - For inspiration on how to code a neural network in Python
- <a href = 'https://www.youtube.com/watch?v=VMj-3S1tku0&t=8065s'>Neural Networks Introduction video</a> by Andrej Karpathy
    - For understanding how neural networks works, and understanding backpropagation
- <a href = 'https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf'>Evolving Neural Networks through Augmenting Topologies
    - The original NEAT paper, which provides me with details about the implementation