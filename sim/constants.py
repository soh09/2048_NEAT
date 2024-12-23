# parameters for mutation chance of NetworkGenome 
NEURON_ADD_CHANCE = 0.001
SYNAPSE_WEIGHT_CHANGE_CHANCE = 0.005
SYNAPSE_ADD_CHANCE = 0.005
SYNAPSE_SWITCH_CHANCE = 0.005

# parameters for NetworkGenome gene distance
W_DISJOINT = 0.5
W_EXCESS = 0.5
W_WEIGHT = 0.1

# simulation parameters
POP_SIZE = 1000
N_GENS = 100
SPECIATION_THRESHOLD = 2
KILL_SPECIES_AFTER_NO_IMPROVEMENTS = 20 # kill species that don't make an improvement after this many generations

# reward types [MAX NUMBER, COMBINED NUMBERS]
# MAX NUMBER = the max number in the board at the time of death
# COMBINED NUMBER = number of squares that were combined during the game

REWARD_TYPE = 'COMBINED NUMBERS'

# game paramters
FOUR_CHANCE = 0.3 # chance that the new number in a game is a 4, instead of 2