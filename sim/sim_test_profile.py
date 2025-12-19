import sys
import os
import time

# Robustly find the parent directory (2048_NEAT)
current_dir = os.path.dirname(os.path.abspath(__file__)) # Path to sim/
parent_dir = os.path.dirname(current_dir)                # Path to 2048_NEAT/

# Insert at position 0 to ensure it takes precedence
sys.path.insert(0, parent_dir)

from sim.Simulation import Simulation
import neural_net.nn as nn
from neural_net.visualize import visualize_genome, visualize_net, display_gene
from Sandbox import Sandbox
import random
import gc

res = []

logging_path = 'C:\\Users\\hirot\Documents\\2048_NEAT\\logs\\actual_logs'
checkpoint_path = 'C:\\Users\\hirot\Documents\\2048_NEAT\\checkpoints'
a = Simulation(log_folder = logging_path, checkpoint_folder = checkpoint_path)


a.simulate()
a.adjust_fitness()
a.reproduce()
a.mutate_and_speciate()

for i in range(2000):

    now = time.time()
    a.simulate()
    sim = time.time() - now

    a.adjust_fitness()

    a.reproduce()

    now = time.time()
    a.mutate_and_speciate()
    mas = time.time() - now
    print(f'mutate: {mas:.3f}, simulate: {sim:.3f}')

    if i % 20 == 0:
        now = time.time()
        gc.collect()
        print(f'time to gc: {time.time() - now}')
    print('--------------------------------')