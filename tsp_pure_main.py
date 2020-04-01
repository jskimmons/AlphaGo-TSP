from tsp.TSPGame import TSPGame as Game
from PureMCTS import PureMCTS
from tqdm import tqdm
import numpy as np
from utils import *

args = dotdict({
    'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
    'numMCTSSims': 30000,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,

})

if __name__ == "__main__":
    game = Game(10)
    actions = [game.start_node]
    wins, losses = 0, 0
    mcts = PureMCTS(game, args)
    player = 1
    for i in range(args.numEps):
        board = game.getInitBoard()
        while game.getGameEnded(board, player) == 0:
            canon = game.getCanonicalForm(board, player)
            ap = mcts.getActionProb(canon, temp=1)
            action = np.argmax(ap)
            actions.append(action)
            valid_moves = game.getValidMoves(canon, player)
            if valid_moves[action] == 0:
                exit(1)
            board, player = game.getNextState(board, player, action)
        print('my', actions)
        actions = [game.start_node]
        if game.getGameEnded(board, player) == 1:
            wins += 1
        else:
            losses += 1

    print('wins', wins)
    print('losses', losses)
