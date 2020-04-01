from unittest import TestCase
from tsp.TSPGame import TSPGame
from tsp.TSPPlayers import RandomPlayer
import numpy as np


class TestTSPGame(TestCase):
    def test_getInitBoard(self):
        nodes = 10
        game = TSPGame(nodes)
        board, cur_len = game.getInitBoard()
        self.assertEqual(board.shape, (3, nodes, nodes))

    def test_getBoardSize(self):
        nodes = 10
        game = TSPGame(nodes)
        _, _ = game.getInitBoard()
        self.assertEqual(game.getBoardSize(), (nodes, nodes))

    def test_getActionSize(self):
        nodes = 10
        game = TSPGame(nodes)
        _, _ = game.getInitBoard()
        self.assertEqual(game.getActionSize(), nodes)

    def test_getNextState(self):
        # assert the board state is changing properly
        nodes = 10
        game = TSPGame(nodes)
        board = game.getInitBoard()
        p = RandomPlayer(game)
        action = 6
        board2, _ = game.getNextState(board, 1, action)

        self.assertTrue(np.array_equal(board[0][0], board2[0][0]))
        self.assertFalse(board[0][1][:, action].any(), 0)
        self.assertTrue(board2[0][1][:,action].all(), 1)
        self.assertEqual(board2[0][2][0][0], action)

    def test_getValidMoves(self):
        nodes = 10
        game = TSPGame(nodes)
        board = game.getInitBoard()
        p = RandomPlayer(game)
        actions = [3, 6, 8]
        for action in actions:
            board, _ = game.getNextState(board, 1, action)

        valid_moves = game.getValidMoves(board, 1)
        self.assertEqual(len(valid_moves), nodes)
        self.assertEqual(sum(valid_moves), nodes - len(actions) - 1)
        for action in actions:
            self.assertEqual(valid_moves[action], 0)

    def test_getGameEnded(self):
        nodes = 10
        game = TSPGame(nodes)
        board = game.getInitBoard()
        p = RandomPlayer(game)
        actions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        checksum = 0
        cur_node = 0
        for action in actions:
            self.assertEqual(game.getGameEnded(board, 1), 0)
            board, _ = game.getNextState(board, 1, action)
            checksum += board[0][0][cur_node][action]
            cur_node = action

        valid_moves = game.getValidMoves(board, 1)
        self.assertEqual(len(valid_moves), nodes)
        self.assertFalse(valid_moves.any())
        self.assertNotEqual(game.getGameEnded(board, 1), 0)
        self.assertEqual(checksum, board[1])
