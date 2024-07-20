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

    def __init__(self, network: 'Network'):
        self.network = network
        self.game = Game()

    def set_input(self):
        # set the input
        self.network.set_input(self.game.get_board())
    
    def make_next_move(self):
        # perform forward pass
        self.network.forward()
        # softmax the output layer
        self.network.output_l.softmax()
        # map max activation to a movement
        max_neuron_idx = max(range(self.network.output_l.n_neurons), key=lambda i: self.network.output_l.neurons[i])
        move = Sandbox.neuron_to_move[max_neuron_idx]

        new_game_state = self.game.make_next_move(move)
        print(self.game)

        if new_game_state == 'lose':
            max_n = self.game.get_max_item()
            self.network.set_fitness(max_n)
            raise GameLostException(f'Game lost at score {max_n}')
        elif new_game_state == 'win':
            self.network.set_fitness(2048)
            raise GameWonException('Game won')

    def reset_update(self):
        self.network.flush_values()
        self.game.generate_next()




class GameLostException(Exception):
    pass

class GameWonException(Exception):
    pass
