# So-NEAT 
Excuse the pun. 

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
    - courtesy of github repo: https://github.com/yangshun/2048-python
    - thank you very much for your code. I will take good care of it.
2. Integrate the 2048 game with the model
    - enable ai model to understand the current game state
        - what numbers there are, how many spaces are left open, etc
        - or just straight up the numbers of each square, since there aren't that many 
    - enable model to control the game
        - simple key inputs (arrow keys most likely)

## The neural network + NEAT component
1. Implement a simple neural network
    - create a neuron class: DONE
    - create a synapse class: DONE
    - implement genotype functionality: do more research
    - implement forward pass capability: DONE
        - "forward pass" should forward the activation of a neuron to connected neuron
    - create a layer class: DONE
        - Layer class will be useful for the input and output layers of the network, since the 
        number of neurons are fixed for these two cases
        - will have softmax capability to generate probabilities of each move
    - ~~create a multi-layer perceptron class~~ not necessary, since the point of
    NEAT is to create flexible topologies
    - implement network class: DONE
        - similar to a MLP, but because of NEAT, doesn't have hidden layers, but
        has hidden neurons instead
        - implement ability to forward pass on hidden neurons: DONE
            - use graph data structure for hidden neurons
            - implement topological search for forward pass capability: DONE

2. Add NEAT functionality to step 1
    - implement crossover (mating) functionality
    - implement building a model (phenotype) from the genotype
    - implement mutation
        - make sure no cycles occur as a result
            - tarjan's strongly connected components alg
            - or find a spot for node, then only look to connect components after it
    - implement some sort of species differentiating algorithm


## Some Technical Considerations
### Neural Network
- Layer class .forward() function will be used for the input layer
- Layer class .softmax() function will be used for the output layer
- for implementing forward pass for hidden neurons, the topological sort will be
performed on the Synapses (the links), not the neurons. 







# Attribution

## 2048-python
Github repo: https://github.com/yangshun/2048-python<br>
Contributors:
- [Yanghun Tay](http://github.com/yangshun)
- [Emmanuel Goh](http://github.com/emman27)

## General Resources
- <a href = 'https://www.youtube.com/watch?v=lAjcH-hCusg'>NEAT algorithm from scratch (it was hard)</a> by Tech with Nikola
    - for inspiration on how to code NEAT
- <a href = 'https://www.youtube.com/watch?v=b3D8jPmcw-g'>Neuroevolution of Augmenting Topologies (NEAT)</a> by Connor Shorten
    - for understanding how NEAT works
- <a href = 'https://github.com/karpathy/micrograd/tree/master'>micrograd neural network Library</a> by Andrej Karpathy
    - for inspiration on how to code a neural network in Python
- <a href = 'https://www.youtube.com/watch?v=VMj-3S1tku0&t=8065s'>Neural Networks Introduction video</a> by Andrej Karpathy
    - for understanding how neural networks works, and understanding backpropagation