# So-NEAT 
Excuse the pun. This is a work in progress.
Start Date: 11/22/2023

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
    - [x] implement some sort of species differentiating algorithm
    - [x] implement logic for evolving networks
        - [ ] when to "kill" certain underperforming species
        - [ ] probabilities for sexual reproduction, asexual reproduction, etc

3. Simulation of Generations and Populations
    - [x] create Sandbox class, which takes in a Network class, instantiates a Game object, and makes it play till game over/win condition is met

4. Optimization
    - [x] enable multiprocessing for parallel simulation
    - [ ] optimize nn.NetworkGenome.distance() 
        - [x] optimized the difference calculation by removing unnecessary set creation, adding early stopping

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

### Optimization
- This is the first time I'm having to actually profile my code to determine which lines are taking up precious runtime. It's been pretty interesting.
- I'm using the kernprof tool, which allows me to profile the performance of each line
    - \# of hits, and total time it took that run that line
#### Optimizing NetworkGenome.distance()
- I'm trying various things like
    - early stopping (if the speciation threshold is met at certain points in the code, return, instead of continuing to the end)
        - this works because for our purposes, we just need to know if they are over or under the threshold, the actual value does not matter
    - removing unnecessary function calls
        - I was creating a set of synapse IDs from a list, which is an O(N) operation. 
        This is necessary, because I have a dict whose key's are the IDs. I can simply do `e in dict` to achieve O(1) set membership.
    - instead of iterating net1.synapse_ids and net2.synapse_ids separately, union them, and iterate over this new set.
        - this approach sounded good, but it made early stopping harder. In the two loops case, I can check if the 
        threshold is met after the first loop (so only one early stopping point), but with this approach, I would have to check 
        if the early stopping condition is met at every iteration. The early stopping condition actually now adds overhead. I could
        maybe do like a check every 100 iterations, but at this point, I went with two loops, early stopping, remove unnecessary set creation. This 
        approach already created enough time savings, imo.

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
|  8/3  | Debugged speciation logic. Need to fix mutate_and_speciate(), resets dictionary prematurely. |
|  8/5  | Debugged Simulation.py. Started testing simulation, seems to be working, might need to implement multithreading for simulation to speed things up |
|  8/15 | My worst fear has been confirmed. Multiprocessing requires me to rethink a lot of my code, as when you pass in an object to a function to be multiprocessed, it creates a COPY of the object, it doesn't use the actual object ;-; |
| 8/16  | Implemented MultiProcessing. It actually wasn't that hard thankfully. |
| 8/18  | Every 3-4 iterations, the code spikes (like x10-20) in runtime, and I need to figure out why. This is majorly contributing to slow training. Profiling code in misc_dev folder. |
| 8/18 (2)  | Profiled code. Don't know about the mutation spiking, but brought down runtime in general by optimizing NetworkGenome.distance(). |
|  9/1  | Optimized mutate_and_speciate(). No longer deepcopying the entire NetworkGenome for progenitor, instead only deepcopying the synapse_gene list. Brought down mutate_and_speciate() runtime by half. |
| 9/1 (2) | Added logic in reproduce() to kill off stagnant species. Added logic to make 3% of population a hybrid of two random species, this will introduce more genetic variation and allow for faster search. |
| 9/21  | Showed 2048 AI project to former boss Sato-san. He really liked it hehe. |
| 9/23  | Added new reward function with returns the number of cubes combined. This is a more fine grain reward signal compared to the highest number. Misc debugging. Feels like something is odd, look into if adjusted fitnesses are being calculated properly. | 
| 12/22 | Picking up where I left off. Modified 2048 code to randomly choose 2 or 4 for new number. Double checked COMBINED NUMBERS fitness function. Added logging functionality.
| 12/23 | Trying to identify root cause of runtime spikes. Pinpointed `simulation()` runtime spikes to `nn.Network()` construction. Will investigate further. Added charts in log/ for simulation performance logging.
| 12/24 | "Network complexity explosion" is not the culprit of simulation() runtime explosion. Need to investigate further.

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
- <a href = 'https://chatgpt.com'>ChatGPT
    - The obvious