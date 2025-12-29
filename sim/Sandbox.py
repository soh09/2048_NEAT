from neural_net.nn import Network
from python_2048.fast2048 import Game
from sim.constants import REWARD_TYPE, INTENT_PENALTY

class Sandbox:
    # neurons in output layer correspond to the following movements
    # [0: up, 1: down, 2: right, 3: left]
    neuron_to_move = {
        0: 'up',
        1: 'down',
        2: 'right',
        3: 'left'
    }

    def __init__(self, network: 'Network', debug = False, reward = REWARD_TYPE, normalize_input = True):
        self.network = network
        self.game = Game(reward)
        self.previous_state = self.game.get_board()
        self.debug = debug
        self.last_move = None
        self.normalize_input = normalize_input
        self.invalid_moves = 0
        self.valid_moves = 0

    def set_input(self):
        # set the input
        if self.normalize_input:
            self.network.set_input([x/15 for x in self.game.get_board()])
        else:
            self.network.set_input(self.game.get_board())

    # @profile
    def make_next_move(self, frozen = False):
        # perform forward pass
        self.network.forward()
        # softmax the output layer
        self.network.output_l.softmax()
        # map max activation to a movement
        moves_sorted_by_activation = sorted(range(self.network.output_l.n_neurons), key=lambda i: self.network.output_l.neurons[i].get_activation(), reverse=True)
        
        made_move = False

        for _, move in enumerate(moves_sorted_by_activation):

            move_str = Sandbox.neuron_to_move[move]
            self.last_move = move_str
            new_game_state = self.game.do_next_move_and_track(move_str, self.debug)

            if new_game_state != 'board unchanged':
                self.valid_moves += 1
                made_move = True

                reward = self.game.get_reward() * (self.valid_moves / (self.invalid_moves + self.valid_moves))
                if not frozen:
                    self.network.set_fitness(reward)
                else:
                    self.network.genome.temp_fitness = reward

                if new_game_state == 'lose':
                    raise GameLostException(f'Game lost at score {reward}')
                elif new_game_state == 'win':
                    raise GameWonException('Game won')
                break # to prevent AI from making 4 moves in a single turn!
            else:
                self.invalid_moves += 1
            
        # if, after doing all moves, we still haven't moved
        if made_move == False:
            reward = self.game.get_reward() * (self.valid_moves / (self.invalid_moves + self.valid_moves))
            if not frozen:
                self.network.set_fitness(reward)
            else:
                self.network.genome.temp_fitness = reward
            raise GameStuckException(f'Game stuck at score {reward}')


    # called after every move made
    def reset_update(self):
        self.network.flush_values()
        self.game.generate_next()
        self.previous_state = self.game.get_board()

    # called once per game, at the end, to reset board state
    def factory_reset(self):
        self.game.reset()
        self.last_move = None
        self.previous_state = self.game.get_board()
        self.valid_moves = 0
        self.invalid_moves = 0


class GameLostException(Exception):
    pass

class GameWonException(Exception):
    pass

class GameStuckException(Exception):
    pass