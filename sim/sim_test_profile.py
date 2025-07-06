import sys
import os
import time

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join('..')))

from Simulation import Simulation
import neural_net.nn as nn
from neural_net.visualize import visualize_genome, visualize_net, display_gene
from Sandbox import Sandbox
import random

res = []

logging_path = '/Users/so/Documents/projects/personal/2048_AI/logs'

now = time.time()
a = Simulation(log_folder = logging_path)


a.simulate()
a.adjust_fitness()
a.reproduce()
# [s.fitness for s in a.species[0]['children']]

for i in range(50):
    now = time.time()
    a.mutate_and_speciate()
    mas = time.time() - now

    now = time.time()
    a.simulate()
    sim = time.time() - now

    a.adjust_fitness()

    a.reproduce()

    print(f'mutate: {mas:.3f}, simulate: {sim:.3f}')

    # if sim >= 10:
    #     break