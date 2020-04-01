import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time

class SinglePlayerArena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.game = game
        self.display = print

    def playSinglePlayerGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        player = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, player)==0:
            it+=1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(player))
                self.display(board)
            action = self.player1(self.game.getCanonicalForm(board, player))
            print(action)
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, player), 1)
            if valids[action]==0:
                print(action)
                assert valids[action] >0
            board, player = self.game.getNextState(board, player, action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
            print('final length: {}'.format(self.game.env.get_weight(board, 0)))
            print('optim length: {}'.format(self.game.env.get_optimal_length()))
        return self.game.getGameEnded(board, player)

    def playSinglePlayerGames(self, num, verbose=False):
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        wins = 0
        losses = 0
        draws = 0
        for _ in range(num):
            gameResult = self.playSinglePlayerGame(verbose=verbose)
            if gameResult == 1:
                wins += 1
            elif gameResult == -1:
                losses += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps,
                                                                                                       maxeps=maxeps,
                                                                                                       et=eps_time.avg,
                                                                                                       total=bar.elapsed_td,
                                                                                                       eta=bar.eta_td)
            bar.next()
        bar.finish()
        return wins, losses, draws