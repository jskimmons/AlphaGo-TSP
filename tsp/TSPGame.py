from __future__ import print_function
from Game import Game
from tsp.TSPLogic import TSPEnviornment

import numpy as np
import random


class TSPGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self, nodes):
        Game.__init__(self)
        self.env = TSPEnviornment(nodes=nodes)
        self.start_node = random.randint(0, nodes-1)

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """

        return self.env.new_board(self.start_node)

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.env.nodes, self.env.nodes

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.env.nodes

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        new_board = self.env.visit_node(board, action)
        return new_board, player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        # this is any move not already seen in the path
        graph, path = board
        valid_moves = np.ones(shape=(self.getActionSize(),), dtype=np.int32)
        for node in path:
            valid_moves[node] = 0
        return valid_moves

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        # if any of these values in the board[1][0] are 0, game is not over
        if sum(self.getValidMoves(board, player)) != 0:
            return 0
        else:
            # calculate the path length
            current_path_length = self.env.get_path_length(board)
            if current_path_length <= (1.1 * self.env.optimal_path_length):
                return 1
            else:
                return -1

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """

        # the path is the string representation
        # since the game board is the same for each mcts instance,
        # path is enough to tell game states apart
        return str(board)
