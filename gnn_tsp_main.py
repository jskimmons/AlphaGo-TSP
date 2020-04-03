from TSPCoach import TSPCoach
from tsp.TSPGame import TSPGame as Game
from tsp.TSPNNet import NNetWrapper as nn
from tsp.NNetShell import NNetShell as shell
from utils import *

args = dotdict({
    'numIters': 1,
    'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'maxlenOfQueue': 1000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 0,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'numItersForTrainExamplesHistory': 2000,
})

if __name__ == "__main__":
    game = Game(10)
    nnet = nn(game)
    c = TSPCoach(args, game, nnet)
    c.learn()

    args = dotdict({
        'numIters': 1,
        'numEps': 0,  # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,  #
        'maxlenOfQueue': 1000,  # Number of game examples to train the neural networks.
        'numMCTSSims': 50,  # Number of games moves for MCTS to simulate.
        'arenaCompare': 0,  # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,
        'numItersForTrainExamplesHistory': 2000,
    })

    game = Game(20)
    nnet = nn(game)
    c = TSPCoach(args, game, nnet)
    c.learn()