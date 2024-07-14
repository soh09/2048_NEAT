from neural_net.nn import Network
from python_2048.twenty_forty_eight import Game

class Sandbox:
    def __init__(self, network: 'Network'):
        self.network = network
        self.game = Game()