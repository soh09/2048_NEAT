from neural_net.nn import Network, NetworkGenome
from sim.Sandbox import Sandbox
from sim.constants import REWARD_TYPE

def print_histories(fitnesses, scores, per_row=10):
    # 1. Calculate the Global Max Width across BOTH lists
    #    (We look at every number in both lists to find the widest one)
    all_data = fitnesses + scores
    if not all_data:
        width = 5 
    else:
        width = max(len(f'{round(x, 3)}') for x in all_data) + 2
    
    # 2. Define a quick helper to print using that shared width
    def _print_sublist(label, data):
        s = label
        for i in range(0, len(data), per_row):
            row = data[i:i+per_row]
            # Use the global 'width' here
            s += "  " + "".join(f"{str(f'{round(x, 3)}'):>{width}}" for x in row)
        print(s)

    # 3. Print both
    _print_sublist("Full Fitness History", fitnesses)
    _print_sublist("Full Scores History ", scores)

def test_genome(obj):

    # 1. Load Genome
    try:
        genome = NetworkGenome.load_genome(obj)
    except TypeError as e:
        try:
            genome = obj
        except Exception as e:
            print('input object is neither a genopme pickle or a genome instance')

    net = Network(genome)
    sandbox = Sandbox(net, reward=REWARD_TYPE)
    print(f'Reward function: {sandbox.game.reward_type}')

    fitnesses = []
    scores = []
    move_hist = []
    samples = 10

    move_dict = {
        'up': '↑',
        'down': '↓',
        'left': '←',
        'right': '→' 
    }

    for _ in range(samples):
        # print(sandbox.game)
        hist = []
        while True:
            try:
                sandbox.set_input()
                sandbox.make_next_move(frozen=True)
                hist.append(move_dict[sandbox.last_move])
                sandbox.reset_update()
            except Exception:
                sandbox.network.clear()
                # Return fitness when game ends
                fitnesses.append(sandbox.network.genome.temp_fitness)
                scores.append(sandbox.game.score)
                # print(sandbox.game)
                sandbox.factory_reset()
                break
        move_hist.append(hist)
    avg_fitness = sum(fitnesses) / samples
    avg_score = sum(scores) / samples
    # Pre-calculate min/max to keep the f-string clean
    min_fit, max_fit = min(fitnesses), max(fitnesses)
    min_scr, max_scr = min(scores), max(scores)
    print(f'Stats from running {samples} simulations')
    print(f'''_________________________________________
| Metric  |   Mean   |   Min   |   Max   |
|_________|__________|_________|_________|
| Fitness | {avg_fitness:8.2f} | {min_fit:7.2f} | {max_fit:7.2f} |
| Score   | {avg_score:8.2f} | {min_scr:7.2f} | {max_scr:7.2f} |
|_________|__________|_________|_________|''')
    print_histories(fitnesses, scores)
    print("\n".join(f"[{''.join(hist)}]" for hist in move_hist))

