# parameters for mutation chance of NetworkGenome 
NEURON_ADD_CHANCE = 0.1
# NEURON_ADD_CHANCE = 1
SYNAPSE_ADD_CHANCE = 0.05
SYNAPSE_WEIGHT_CHANGE_CHANCE = 0.8
# SYNAPSE_WEIGHT_CHANGE_CHANCE = 0
SYNAPSE_SWITCH_CHANCE = 0.05
# SYNAPSE_SWITCH_CHANCE = 0.5
SYNAPSE_WEIGHT_PERTERB_CHANCE = 0.9 # chance that we do a gaussian perterb, else we do a random weight sampling

# parameters for NetworkGenome gene distance
W_DISJOINT = 1.5
W_EXCESS = 1.5
W_WEIGHT = 0.3

# simulation parameters
POP_SIZE = 1000
# N_GENS = 100
SPECIATION_THRESHOLD = 0.75
KILL_SPECIES_AFTER_NO_IMPROVEMENTS = 15 # kill species that don't make an improvement after this many generations
SAMPLES = 5

# reward types [MAX NUMBER, SCIRE, TILES COMBINED, SCORE + EMPTY TILES BONUS]
# MAX NUMBER = the max number in the board at the time of death
# COMBINED NUMBER = number of squares that were combined during the game

# REWARD_TYPE = 'SCORE'
REWARD_TYPE = 'SCORE + EMPTY TILES BONUS'
INTENT_PENALTY = 0.99

# game paramters
FOUR_CHANCE = 0.3 # chance that the new number in a game is a 4, instead of 2