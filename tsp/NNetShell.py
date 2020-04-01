from NeuralNet import NeuralNet

import time
import numpy as np
import sys


class NNetShell(NeuralNet):

    def __init__(self, game):
        self.game = game
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        pass

    def predict(self, board):
        """
        board: np array with board
        """
        valid_moves = self.game.getValidMoves(board, 1)
        pi = valid_moves / sum(valid_moves)
        return pi, 0

    def save_checkpoint(self, folder, filename):
        pass

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        pass