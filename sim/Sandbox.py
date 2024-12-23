from neural_net.nn import Network
from python_2048.twenty_forty_eight import Game

class Sandbox:
    # neurons in output layer correspond to the following movements
    # [0: up, 1: down, 2: right, 3: left]
    neuron_to_move = {
        0: 'up',
        1: 'down',
        2: 'right',
        3: 'left'
    }

    def __init__(self, network: 'Network', debug = False):
        self.network = network
        self.game = Game()
        self.previous_state = self.game.get_board()
        self.debug = debug

    def set_input(self):
        # set the input
        self.network.set_input(self.game.get_board())
    
    def make_next_move(self, reward_type):
        # perform forward pass
        self.network.forward()
        # softmax the output layer
        self.network.output_l.softmax()
        # map max activation to a movement
        max_neuron_idx = max(range(self.network.output_l.n_neurons), key=lambda i: self.network.output_l.neurons[i].get_activation())
        move = Sandbox.neuron_to_move[max_neuron_idx]

        if self.debug:
            print(self.game)
        new_game_state = self.game.do_next_move_and_track(move, self.debug)
        if self.debug:
            print(self.game)

        if self.game.get_board() == self.previous_state:
            reward = self.game.get_reward(reward_type)
            self.network.set_fitness(reward)
            # print(self.game)
            raise GameStuckException(f'Game stuck at score {reward}')
        if new_game_state == 'lose':
            reward = self.game.get_reward(reward_type)
            self.network.set_fitness(reward)
            # print(self.game)
            raise GameLostException(f'Game lost at score {reward}')
        elif new_game_state == 'win':
            self.network.set_fitness(2048)
            # print(self.game)
            raise GameWonException('Game won')


    def reset_update(self):
        self.network.flush_values()
        self.game.generate_next()
        self.previous_state = self.game.get_board()




class GameLostException(Exception):
    pass

class GameWonException(Exception):
    pass

class GameStuckException(Exception):
    pass