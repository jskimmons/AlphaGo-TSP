from tsp.TSPGame import TSPGame
from tsp.TSPLogic import TSPEnviornment
from pprint import pprint


if __name__ == '__main__':
    game = TSPGame(3)
    board = game.getInitBoard()
    board2, player = game.getNextState(board, 1, 2)
    pprint(board2)


