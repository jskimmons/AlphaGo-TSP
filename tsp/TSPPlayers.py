import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valid_moves = self.game.getValidMoves(board, 1)
        while valid_moves[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a
